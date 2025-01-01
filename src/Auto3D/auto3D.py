#!/usr/bin/env python
"""
Generating low-energy conformers from SMILES.
"""
import glob
import logging
import math
import multiprocessing as mp
import os
import shutil
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from logging.handlers import QueueHandler
from typing import List, Optional, Union

import pandas as pd
import psutil
import torch
from rdkit import Chem
from send2trash import send2trash

import Auto3D
from Auto3D.batch_opt.batchopt import optimizing
from Auto3D.isomer_engine import (
    generate_isomers_as_sdf,
    handle_tautomers,
    oe_isomer,
    rd_isomer,
    rd_isomer_sdf,
)
from Auto3D.ranking import ranking
from Auto3D.utils import check_input, create_chunk_meta_names, housekeeping, reorder_sdf
from Auto3D.utils_file import SDF2chunks, decode_ids, encode_ids, smiles2smi

try:
    mp.set_start_method("spawn")
except:
    pass

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _prepare_logger(logging_queue):
    logger = logging.getLogger("auto3d")
    logger.addHandler(QueueHandler(logging_queue))
    logger.setLevel(logging.INFO)
    return logger


def _clean_up_intermediate_files(directory, housekeeping_folder, output, verbose: bool):
    r"""
    Zip all meta data if verbose is True.

    Arguments:
        directory: the folder to be cleaned up
        housekeeping_folder: a folder to contain all meta data
        output: the output file that will not be moved
    """
    # Housekeeping
    os.mkdir(housekeeping_folder)
    housekeeping(directory, housekeeping_folder, output)
    # Compress verbose folder
    housekeeping_folder_gz = housekeeping_folder + ".tar.gz"
    with tarfile.open(housekeeping_folder_gz, "w:gz") as tar:
        tar.add(housekeeping_folder, arcname=os.path.basename(housekeeping_folder))
    shutil.rmtree(housekeeping_folder)
    if not verbose:
        try:  # Clusters do not support send2trash
            send2trash(housekeeping_folder_gz)
        except:
            os.remove(housekeeping_folder_gz)


def isomer_wraper(chunk_info, config, queue, logging_queue):
    """
    chunk_info: (path, dir) tuple for the chunk
    config: auto3D arguments
    queue: mp.queue
    logging_queue
    """
    logger = _prepare_logger(logging_queue)

    for i, path_dir in enumerate(chunk_info):
        print(f"\n\nIsomer generation for job{i+1}", flush=True)
        logger.info(f"\n\nIsomer generation for job{i+1}")
        path, directory = path_dir
        meta = create_chunk_meta_names(path, directory)
        # Tautomer enumeratioin
        if config.enumerate_tautomer:
            path = handle_tautomers(path=path, meta=meta, config=config, logger=logger)
        enumerated_sdf = generate_isomers_as_sdf(
            path=path, directory=directory, meta=meta, config=config
        )
        queue.put((enumerated_sdf, path, directory, i + 1))
    if isinstance(config.gpu_idx, int) or len(config.gpu_idx) == 1:
        queue.put("Done")
    else:
        for _ in range(len(config.gpu_idx)):
            queue.put("Done")


def optim_rank_wrapper(
    config, queue, logging_queue, device: torch.device, do_ranking: bool = True
) -> List[Chem.Mol]:

    logger = _prepare_logger(logging_queue)

    while True:
        sdf_path_dir_job = queue.get()
        if sdf_path_dir_job == "Done":
            break
        enumerated_sdf, path, directory, job = sdf_path_dir_job
        print(f"\n\nOptimizing conformers for job{job}", flush=True)
        logger.info(f"\n\nOptimizing conformers for job{job}")
        meta = create_chunk_meta_names(path, directory)

        # Optimizing step
        opt_config = {
            "opt_steps": config.opt_steps,
            "opttol": config.convergence_threshold,
            "patience": config.patience,
            "batchsize_atoms": config.batchsize_atoms,
        }
        optimizer = optimizing(
            in_f=enumerated_sdf,
            out_f=meta["optimized_og"],
            name=config.optimizing_engine,
            device=device,
            config=opt_config,
        )
        optimizer.run()

        # Ranking step
        if do_ranking:
            rank_engine = ranking(
                input_path=meta["optimized_og"],
                out_path=meta["output"],
                threshold=config.threshold,
                k=config.k,
                window=config.window,
                encoded=True,
            )
            rank_engine.run()
        else:
            shutil.move(meta["optimized_og"], meta["output"])

        _clean_up_intermediate_files(
            directory=directory,
            housekeeping_folder=meta["housekeeping_folder"],
            output=meta["output"],
            verbose=config.verbose,
        )


@dataclass
class Config:
    r"""Generate configuration arguments for the Auto3D generate_and_optimize_conformers method.

    Args:
        input_file (str, optional): Path to input.smi file containing SMILES and IDs. See example/files folder for examples.
        output_file (str, optional): Path to the output file. Should be an sdf file. If specified and verbose=False, the job folder will be deleted.

        # Output Control
        k (bool, optional): Number of conformers for each molecule. Defaults to None.
        window (bool, optional): Whether to output structures with energies within x kcal/mol from the lowest energy conformer. Defaults to None.
        job_name (str, optional): Name of folder to save metadata. Defaults to "".
        verbose (bool, optional): Whether to save all metadata during execution. Defaults to False.
        do_ranking (bool, optional): Whether to rank and filter the conformers based on k and window. Defaults to True.

        # Tautomer Settings
        enumerate_tautomer (bool, optional): Whether to enumerate tautomers for input. Defaults to False.
        tauto_engine (str, optional): Program for tautomer enumeration ('rdkit' or 'oechem'). Defaults to "rdkit".
        pKaNorm (bool, optional): Whether to assign predominant ionization state at pH ~7.4. Only works with tauto_engine='oechem'. Defaults to True.

        # Isomer Generation
        isomer_engine (str, optional): Program for 3D isomer generation ('rdkit' or 'omega'). Defaults to "rdkit".
        enumerate_isomer (bool, optional): Whether to enumerate cis/trans and R/S isomers. Defaults to True.
        mode_oe (str, optional): Omega program mode ('classic', 'macrocycle', 'dense', 'pose', 'rocs', 'fast_rocs').
                                See https://docs.eyesopen.com/applications/omega/omega/omega_overview.html. Defaults to "classic".
        mpi_np (int, optional): Number of CPU cores for isomer generation. Defaults to 4.
        max_confs (int, optional): Maximum number of isomers per SMILES. If None, uses (number of heavy atoms - 1). Defaults to None.

        # Hardware Settings
        use_gpu (bool, optional): Whether to use GPU when available. Defaults to True.
        gpu_idx (int or List[int], optional): GPU index(es) to use. Only applies when use_gpu=True. Defaults to 0.
        capacity (int, optional): For generate_and_optimize_conformers and generate_conformers: Number of SMILES to process per 1GB memory. Defaults to 42.
                                For optimize_conformers: Number of molecules to process per 1GB memory. Defaults to 8192.
        memory (int, optional): RAM allocation for Auto3D in GB. Defaults to None.
        batchsize_atoms (int, optional): Number of atoms per optimization batch per 1GB. Defaults to 1024.

        # Optimization Parameters
        optimizing_engine (str, optional): Energy calculation and geometry optimization engine.
                                        Choose from 'ANI2x', 'ANI2xt', 'AIMNET' or path to custom NNP. Defaults to "AIMNET".
        patience (int, optional): Maximum consecutive steps without force decrease before termination. Defaults to 1000.
        opt_steps (int, optional): Maximum optimization steps per structure. Defaults to 5000.
        convergence_threshold (float, optional): Maximum force threshold for convergence. Defaults to 0.003.
        threshold (float, optional): RMSD threshold for considering conformers as duplicates. Defaults to 0.3.
    """

    input_file: Optional[str] = None
    output_file: Optional[str] = None
    k: Optional[int] = None
    window: Optional[float] = None
    verbose: bool = False
    job_name: str = ""
    enumerate_tautomer: bool = False
    tauto_engine: str = "rdkit"
    pKaNorm: bool = True
    isomer_engine: str = "rdkit"
    enumerate_isomer: bool = True
    mode_oe: str = "classic"
    mpi_np: int = 4
    max_confs: Optional[int] = None
    use_gpu: bool = True
    gpu_idx: Union[int, List[int]] = 0
    capacity: int = 42
    optimizing_engine: str = "AIMNET"
    patience: int = 1000
    opt_steps: int = 5000
    convergence_threshold: float = 0.003
    threshold: float = 0.3
    do_ranking: bool = True
    memory: Optional[int] = None
    batchsize_atoms: int = 1024

    def __post_init__(self):
        r"""
        Normalize some of the string inputs to lowercase.
        """
        self.tauto_engine = self.tauto_engine.lower()
        self.isomer_engine = self.isomer_engine.lower()
        self.mode_oe = self.mode_oe.lower()


_allowed_args = list(Config.__annotations__.keys())


def logger_process(queue, logging_path):
    """A child process for logging all information from other processes"""
    logger = logging.getLogger("auto3d")
    logger.addHandler(logging.FileHandler(logging_path))
    logger.setLevel(logging.INFO)
    while True:
        message = queue.get()
        if message is None:
            break
        logger.handle(message)


def create_config(**kwargs):
    for key in kwargs:
        if key not in _allowed_args:
            raise ValueError(
                f"Unknown parameter: {key}. Has to be one of {_allowed_args}."
            )
    return Config(**kwargs)


def _prep_work(**kwargs):

    r"""
    Args:
        config

    Returns:
        job_name: the name of the job
        path0: the path of the _encoded.smi file with molecules and their encoded IDs
        mapping: dictionary of the format {original molecule id: encoded molecule id}
        chunk_line: a mp.Queue for communication between processes, it receives the path of the enumerated sdf file

    """
    # Make sure the spawn method is used
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass

    config = create_config(**kwargs)

    # Some initial checks
    if config.input_file is None:
        sys.exit("Please specify the input file path.")
    path0, mapping = encode_ids(config.input_file)
    input_format = os.path.splitext(path0)[1][1:]
    if (input_format != "smi") and (input_format != "sdf"):
        sys.exit(
            f"Input file type is not supported. Only .smi and .sdf are supported. But the input file is {input_format}."
        )
    config.input_format = input_format
    if config.k is None and config.window is None:
        sys.exit(
            "Either k or window needs to be specified. "
            "Usually, setting '--k=1' satisfies most needs."
        )
    if config.output_file is not None:
        if not config.output_file.endswith(".sdf"):
            sys.exit("Warning: output_file should have a .sdf extension")

    # Create job_name based on the current time
    if config.job_name == "":
        config.job_name = datetime.now().strftime(
            "%Y%m%d-%H%M%S-%f"
        )  # adds microsecond at the end
    job_name = config.job_name

    # A queue managing two wrappers
    chunk_line = mp.Manager().Queue()

    # initialiazation
    basename = os.path.basename(path0)
    dir = os.path.dirname(os.path.abspath(path0))
    job_name = basename.split(".")[0].strip()[:-8] + "_" + job_name  # remove '_encoded'
    job_name = os.path.join(dir, job_name)
    os.mkdir(job_name)

    # initialize the logging process
    logging_path = os.path.join(job_name, "Auto3D.log")
    logging_queue = mp.Manager().Queue(999)
    logger_p = mp.Process(
        target=logger_process, args=(logging_queue, logging_path), daemon=True
    )
    logger_p.start()

    # logger in the main process
    logger = _prepare_logger(logging_queue)
    logger.info(
        rf"""
         _              _             _____   ____
        / \     _   _  | |_    ___   |___ /  |  _ \
       / _ \   | | | | | __|  / _ \    |_ \  | | | |
      / ___ \  | |_| | | |_  | (_) |  ___) | | |_| |
     /_/   \_\  \__,_|  \__|  \___/  |____/  |____/  {Auto3D.__version__}
              // Generating low-energy 3D structures"""
    )

    logger.info(
        "================================================================================"
    )
    logger.info("                               INPUT PARAMETERS")
    logger.info(
        "================================================================================"
    )
    for key, val in asdict(config).items():
        line = str(key) + ": " + str(val)
        logger.info(line)

    logger.info(
        "================================================================================"
    )
    logger.info("                               RUNNING PROCESS")
    logger.info(
        "================================================================================"
    )

    check_input(config)

    return config, job_name, path0, mapping, chunk_line, logger, logging_queue


def _divide_jobs_based_on_memory(config):
    # Allow 42 SMILES strings per GB memory by default for generate_and_optimize_conformers
    smiles_per_G = config.capacity
    num_jobs = 1
    if config.memory is not None:
        t = int(config.memory)
    else:
        if config.use_gpu:
            if isinstance(config.gpu_idx, int):
                gpu_idx = config.gpu_idx
            else:
                gpu_idx = config.gpu_idx[0]
                num_jobs = len(config.gpu_idx)
            t = int(
                math.ceil(
                    torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3)
                )
            )
        else:
            t = int(psutil.virtual_memory().total / (1024**3))
    chunk_size = t * smiles_per_G
    # batchsize_atoms based on GPU memory
    config.t = t
    config.batchsize_atoms = config.batchsize_atoms * t
    config.num_jobs, config.chunk_size = num_jobs, chunk_size
    return config


def _save_chunks(config, logger, job_name, path0):
    r"""

    Returns:
        chunk_info: a list of (path, dir) tuples for each chunk (.smi file)
    """

    basename = os.path.basename(path0).split(".")[0].strip()
    input_format = config.input_format
    t, chunk_size, num_jobs = config.t, config.chunk_size, config.num_jobs
    # Get indexes for each chunk
    match input_format:
        case "smi":
            df = pd.read_csv(path0, sep=r"\s+", header=None)
        case "sdf":
            df = SDF2chunks(path0)
    data_size = len(df)
    num_chunks = max(round(data_size // chunk_size), num_jobs)
    print(f"The available memory is {t} GB.", flush=True)
    print(f"The task will be divided into {num_chunks} job(s).", flush=True)
    logger.info(f"The available memory is {t} GB.")
    logger.info(f"The task will be divided into {num_chunks} jobs.")
    chunk_idx = [[] for _ in range(num_chunks)]
    for i in range(num_chunks):
        idx = i
        while idx < data_size:
            chunk_idx[i].append(idx)
            idx += num_chunks
    # Save each chunk as an individual file
    chunk_info = []
    for i in range(num_chunks):
        dir_ = os.path.join(job_name, f"job{i+1}")
        os.mkdir(dir_)
        new_basename = basename + "_" + str(i + 1) + f".{input_format}"
        new_name = os.path.join(dir_, new_basename)
        match input_format:
            case "smi":
                df_i = df.iloc[chunk_idx[i], :]
                df_i.to_csv(new_name, header=None, index=None, sep=" ")
            case "sdf":
                df_i = [df[j] for j in chunk_idx[i]]
                with open(new_name, "w") as f:
                    for chunk in df_i:
                        for line in chunk:
                            f.write(line)
        path = new_name
        print(f"Job{i+1}, number of inputs: {len(df_i)}", flush=True)
        logger.info(f"Job{i+1}, number of inputs: {len(df_i)}")
        chunk_info.append((path, dir_))
    return chunk_info


def _combine_sdfs(job_name, path0, input_suffix="_3d.sdf", output_suffix="_out.sdf"):
    # Combine jobs into a single sdf
    data = []
    paths = os.path.join(job_name, f"job*/*{input_suffix}")
    files = glob.glob(paths)
    if len(files) == 0:
        msg = """The optimization engine did not run, or no 3D structure converged.
                 The reason might be one of the following:
                 1. Allocated memory is not enough;
                 2. The input SMILES encodes invalid chemical structures;
                 3. Patience is too small."""
        sys.exit(msg)
    for file in files:
        with open(file, "r") as f:
            data_i = f.readlines()
        data += data_i
    basename = os.path.basename(path0).split(".")[0].strip()
    combined_basename = basename + output_suffix
    path_combined = os.path.join(job_name, combined_basename)
    with open(path_combined, "w+") as f:
        for line in data:
            f.write(line)
    return path_combined


def _create_and_run_isomer_gen_process(chunk_info, config, chunk_line, logging_queue):
    p1 = mp.Process(
        target=isomer_wraper,
        kwargs={
            "chunk_info": chunk_info,
            "config": config,
            "queue": chunk_line,
            "logging_queue": logging_queue,
        },
    )
    p1.start()
    return p1


def _create_and_run_opt_processes(config, chunk_line, logging_queue, do_ranking):
    p2s = []
    for idx in config.gpu_idx:
        if config.use_gpu:
            device = torch.device(f"cuda:{idx}")
        else:
            device = torch.device("cpu")
        p2s.append(
            mp.Process(
                target=optim_rank_wrapper,
                kwargs={
                    "config": config,
                    "queue": chunk_line,
                    "logging_queue": logging_queue,
                    "device": device,
                    "do_ranking": do_ranking,
                },
            )
        )
    for p2 in p2s:
        p2.start()
    return p2s


def _print_timing(start, end, logger):
    print("Energy unit: Hartree if implicit.", flush=True)
    logger.info("Energy unit: Hartree if implicit.")
    running_time_m = int((end - start) / 60)
    if running_time_m <= 60:
        print(f"Program running time: {running_time_m + 1} minute(s)", flush=True)
        logger.info(f"Program running time: {running_time_m + 1} minute(s)")
    else:
        running_time_h = running_time_m // 60
        remaining_minutes = running_time_m - running_time_h * 60
        print(
            f"Program running time: {running_time_h} hour(s) and {remaining_minutes} minute(s)",
            flush=True,
        )
        logger.info(
            f"Program running time: {running_time_h} hour(s) and {remaining_minutes} minute(s)"
        )


def _clean_up(path_output, logger, logging_queue, job_name, output_file, verbose):
    if output_file is not None:
        shutil.move(path_output, output_file)
        path_output = output_file
    if not verbose:
        for file in glob.glob(f"{job_name}/*"):
            if file != path_output and not os.path.isdir(file):
                os.remove(file)
        try:
            shutil.rmtree(job_name)
        except:
            print(f"Failed to remove the following folder: {job_name}.", flush=True)

    print(f"Output path: {path_output}", flush=True)
    logger.info(f"Output path: {path_output}")
    logging_queue.put(None)
    time.sleep(3)  # wait for the daemon process for 3 seconds
    return path_output


def generate_and_optimize_conformers(**kwargs):

    # Preprocessing work
    config, job_name, path0, mapping, chunk_line, logger, logging_queue = _prep_work(
        **kwargs
    )
    config = _divide_jobs_based_on_memory(config)
    if isinstance(config.gpu_idx, int):
        config.gpu_idx = [config.gpu_idx]

    # Save the initial .smi files as divided above
    chunk_info = _save_chunks(config, logger, job_name, path0)

    start = time.time()

    # Starting the processes
    p1 = _create_and_run_isomer_gen_process(
        chunk_info=chunk_info,
        config=config,
        chunk_line=chunk_line,
        logging_queue=logging_queue,
    )
    p2s = _create_and_run_opt_processes(
        config=config,
        chunk_line=chunk_line,
        logging_queue=logging_queue,
        do_ranking=config.do_ranking,
    )
    p1.join()
    for p2 in p2s:
        p2.join()

    # Combine output files from the jobs into a single sdf
    path_combined = _combine_sdfs(job_name, path0)
    reorder_sdf(sdf=path_combined, source=path0)
    path_output = decode_ids(path=path_combined, mapping=mapping)
    path_output = _clean_up(
        path_output,
        logger,
        logging_queue,
        job_name,
        output_file=config.output_file,
        verbose=config.verbose,
    )
    # Program ends
    end = time.time()
    _print_timing(start, end, logger)
    return path_output


def generate_conformers(**kwargs):
    r"""
    Generate initial conformers from SMILES. Two steps: isomer generation and initial conformer embedding.
    """

    # Preprocessing work
    kwargs["k"] = 1  # a workaround to avoid the exit message regarding k and window
    kwargs[
        "use_gpu"
    ] = False  # the process of generating isomers and embedding conformers does not utilize GPUs.
    config, job_name, path0, mapping, chunk_line, logger, logging_queue = _prep_work(
        **kwargs
    )
    config = _divide_jobs_based_on_memory(config)
    chunk_info = _save_chunks(config, logger, job_name, path0)
    start = time.time()

    # Starting the process
    p1 = _create_and_run_isomer_gen_process(
        chunk_info=chunk_info,
        config=config,
        chunk_line=chunk_line,
        logging_queue=logging_queue,
    )
    p1.join()
    path_combined = _combine_sdfs(
        job_name, path0, input_suffix="_enumerated.sdf", output_suffix="_conformers.sdf"
    )
    reorder_sdf(sdf=path_combined, source=path0, clean_suffix=True)
    path_output = decode_ids(path=path_combined, mapping=mapping, suffix="_conformers")
    for _, directory in chunk_info:
        housekeeping_folder = os.path.join(directory, "intermediate_files")
        _clean_up_intermediate_files(
            directory=directory,
            housekeeping_folder=housekeeping_folder,
            verbose=config.verbose,
            output="",
        )
        if not config.verbose:
            shutil.rmtree(directory)
    path_output = _clean_up(
        path_output,
        logger,
        logging_queue,
        job_name,
        output_file=config.output_file,
        verbose=config.verbose,
    )

    # Program ends
    end = time.time()
    _print_timing(start, end, logger)

    return path_output


def optimize_conformers(**kwargs):
    r"""
    Optimize conformers generated in an sdf file.
    """

    max_conformers_per_GB_memory = 8192

    kwargs[
        "enumerate_isomer"
    ] = False  # to bypass the warning message in check_sdf_format
    if (
        "capacity" not in kwargs
    ):  # default value of number of molecules to process per 1GB memory
        kwargs["capacity"] = max_conformers_per_GB_memory

    config, job_name, path0, mapping, chunk_line, logger, logging_queue = _prep_work(
        **kwargs
    )
    if not config.input_file.endswith(".sdf"):
        sys.exit("Please provide an sdf file for optimization.")
    if config.capacity < max_conformers_per_GB_memory:
        print(
            f"The capacity (number of molecules per 1GB memory) is too small. Please set it to a value >= {max_conformers_per_GB_memory}.",
            flush=True,
        )
        logger.warning(
            f"The capacity (number of molecules per 1GB memory) is too small. Please set it to a value >= {max_conformers_per_GB_memory}."
        )
    config = _divide_jobs_based_on_memory(config)
    if isinstance(config.gpu_idx, int):
        config.gpu_idx = [config.gpu_idx]
    start = time.time()

    # Save the initial .sdf files as divided above
    chunk_info = _save_chunks(config, logger, job_name, path0)

    # This is not pretty, but a quick fix following the queueing logic by the original code for Auto3D
    for i, path_dir in enumerate(chunk_info):
        path, directory = path_dir
        chunk_line.put((path, path, directory, i + 1))
    for _ in range(len(config.gpu_idx)):
        chunk_line.put("Done")

    p2s = _create_and_run_opt_processes(
        config=config,
        chunk_line=chunk_line,
        logging_queue=logging_queue,
        do_ranking=False,
    )
    for p2 in p2s:
        p2.join()

    # Combine output files from the jobs into a single sdf
    path_combined = _combine_sdfs(job_name, path0)
    reorder_sdf(sdf=path_combined, source=path0)
    path_output = decode_ids(path=path_combined, mapping=mapping, suffix="_optimized")

    # Ranking step
    if config.do_ranking:
        rank_engine = ranking(
            input_path=path_output,
            out_path=path_output,
            threshold=config.threshold,
            k=config.k,
            window=config.window,
            encoded=False,
        )
        rank_engine.run()

    path_output = _clean_up(
        path_output,
        logger,
        logging_queue,
        job_name,
        output_file=config.output_file,
        verbose=config.verbose,
    )

    # Program ends
    end = time.time()
    _print_timing(start, end, logger)

    return path_output


def smiles2mols(smiles: List[str], **kwargs) -> List[Chem.Mol]:
    """
    A handy tool for finding the low-energy conformers for a list of SMILES.
    Compared with the ``generate_and_optimize_conformers`` function, it sacrifices efficiency for convenience.
    because ``smiles2mols`` uses only 1 process.
    Both the input and output are returned as variables within Python.

    It's recommended only when the number of SMILES is less than 150;
    Otherwise using the generate_and_optimize_conformers function will be faster.

    :param smiles: A list of SMILES strings for which to find low-energy conformers.
    :type smiles: List[str]
    :param args: A dictionary of arguments as returned by the ``option`` function.
    :type args: dict
    :return: A list of RDKit Mol objects representing the low-energy conformers of the input SMILES.
    :rtype: List[Chem.Mol]
    """
    config = create_config(**kwargs)
    with tempfile.TemporaryDirectory() as tmpdirname:
        basename = "smiles.smi"
        path0 = os.path.join(tmpdirname, basename)
        smiles2smi(smiles, path0)  # save all SMILES into a smi file
        config.input_file = path0
        k = config.k
        window = config.window
        if (not k) and (not window):
            sys.exit(
                "Either k or window needs to be specified. "
                "Usually, setting '--k=1' satisfies most needs."
            )
        config.input_format = "smi"
        check_input(config)

        # smi to sdf
        meta = create_chunk_meta_names(path0, tmpdirname)
        isomer_engine = rd_isomer(
            smi=path0,
            smiles_enumerated=meta["smiles_enumerated"],
            smiles_enumerated_reduced=meta["smiles_reduced"],
            smiles_hashed=meta["smiles_hashed"],
            enumerated_sdf=meta["enumerated_sdf"],
            job_name=tmpdirname,
            max_confs=config.max_confs,
            threshold=0.03,
            np=config.mpi_np,
            flipper=config.enumerate_isomer,
        )
        isomer_engine.run()

        # optimize conformers
        if config.use_gpu:
            if isinstance(config.gpu_idx, int):
                idx = config.gpu_idx
            else:
                idx = config.gpu_idx[0]
            device = torch.device(f"cuda:{idx}")
        else:
            device = torch.device("cpu")
        opt_config = {
            "opt_steps": config.opt_steps,
            "opttol": config.convergence_threshold,
            "patience": config.patience,
            "batchsize_atoms": config.batchsize_atoms,
        }
        opt_engine = optimizing(
            in_f=meta["enumerated_sdf"],
            out_f=meta["optimized_og"],
            name=config.optimizing_engine,
            device=device,
            config=opt_config,
        )
        opt_engine.run()

        # Ranking step
        rank_engine = ranking(
            input_path=meta["optimized_og"],
            out_path=meta["output"],
            threshold=config.threshold,
            k=k,
            window=window,
        )
        rank_engine.run()
        conformers = reorder_sdf(sdf=meta["output"], source=path0)

        print("Energy unit: Hartree if implicit.", flush=True)
    return conformers
