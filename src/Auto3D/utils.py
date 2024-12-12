#!/usr/bin/env python
"""
Providing utilities for the workflow package.
"""
import collections
import glob
import logging
import math
import os
import re
import shutil
import sys
import warnings
from collections import OrderedDict, defaultdict
from io import StringIO
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import inchi, rdDetermineBonds, rdMolAlign, rdMolTransforms
from rdkit.Chem.rdMolDescriptors import (
    CalcNumAtomStereoCenters,
    CalcNumUnspecifiedAtomStereoCenters,
)
from tqdm.auto import tqdm

from Auto3D.utils_file import guess_file_type

# CODATA 2018 energy conversion factor
hartree2ev = 27.211386245988
hartree2kcalpermol = 627.50947337481
ev2kcalpermol = 23.060547830619026

logger = logging.getLogger("auto3d")


def create_chunk_meta_names(path, dir):
    """Output name is based on chunk input path and directory
    path: chunck input smi path
    dir: chunck job folder
    """
    dct = {}
    output_name = os.path.basename(path).split(".")[0].strip() + "_3d.sdf"
    output = os.path.join(dir, output_name)
    optimized_og = os.path.join(dir, os.path.basename(output).split(".")[0] + "0.sdf")

    output_taut = os.path.join(dir, "smi_taut.smi")
    smiles_enumerated = os.path.join(dir, "smiles_enumerated.smi")
    smiles_reduced = os.path.join(
        dir, os.path.basename(smiles_enumerated).split(".")[0] + "_reduced.smi"
    )
    smiles_hashed = os.path.join(dir, "smiles_enumerated_hashed.smi")
    enumerated_sdf = os.path.join(dir, "smiles_enumerated.sdf")
    sorted_sdf = os.path.join(dir, "enumerated_sorted.sdf")
    housekeeping_folder = os.path.join(dir, "verbose")
    # dct["output_name"] = output_name
    dct["output"] = output
    dct["optimized_og"] = optimized_og
    dct["output_taut"] = output_taut
    dct["smiles_enumerated"] = smiles_enumerated
    dct["smiles_reduced"] = smiles_reduced
    dct["smiles_hashed"] = smiles_hashed
    dct["enumerated_sdf"] = enumerated_sdf
    dct["sorted_sdf"] = sorted_sdf
    dct["housekeeping_folder"] = housekeeping_folder
    dct["path"] = path
    dct["dir"] = dir
    return dct


def check_input(args):
    """
    Check the input file and give recommendations.

    Arguments:
        args: Arguments to auto3d.

    Returns:
        This function checks the format of the input file, the properties for
        each SMILES in the input file.
    """
    print("Checking input file...", flush=True)
    logger.info("Checking input file...")
    # ANI_elements = {1, 6, 7, 8, 9, 16, 17}
    # ANI = True
    # Check --use_gpu
    gpu_flag = args.use_gpu
    if gpu_flag:
        if torch.cuda.is_available() == False:
            sys.exit("No cuda device was detected. Please set --use_gpu=False.")
    isomer_engine = args.isomer_engine
    if ("OE_LICENSE" not in os.environ) and (isomer_engine == "omega"):
        sys.exit(
            "Omega is used as the isomer engine, but OE_LICENSE is not detected. Please use rdkit."
        )
    # Check the installation for open toolkits, torchani
    if args.isomer_engine == "omega":
        try:
            from openeye import oechem
        except:
            sys.exit(
                "Omega is used as isomer engine, but openeye toolkits are not installed."
            )
    if args.optimizing_engine == "ANI2x":
        try:
            import torchani
        except:
            sys.exit(
                "ANI2x is used as optimizing engine, but TorchANI is not installed."
            )
    if os.path.exists(args.optimizing_engine):
        try:
            model_ = torch.jit.load(args.optimizing_engine)
        except:
            sys.exit(
                "A path to a user NNP is used as optimizing engine, but it cannot be loaded by torch.load. See this link for information about saving and loading models: https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model"
            )
    if int(args.opt_steps) < 10:
        sys.exit(
            f"Number of optimization steps cannot be smaller than 10, but received {args.opt_steps}"
        )

    # Check the input format
    if args.input_format == "smi":
        ANI, only_aimnet_smiles = check_smi_format(args)
    elif args.input_format == "sdf":
        ANI, only_aimnet_smiles = check_sdf_format(args)

    print("Suggestions for choosing isomer_engine and optimizing_engine: ", flush=True)
    logger.info(f"Suggestions for choosing isomer_engine and optimizing_engine: ")
    if ANI:
        print(
            "\tIsomer engine options: RDKit and Omega.\n"
            "\tOptimizing engine options: ANI2x, ANI2xt, AIMNET or your own NNP.",
            flush=True,
        )
        logger.info("\tIsomer engine options: RDKit and Omega.")
        logger.info(
            "\tOptimizing engine options: ANI2x, ANI2xt, AIMNET or your own NNP."
        )
    else:
        print(
            "\tIsomer engine options: RDKit and Omega.\n"
            "\tOptimizing engine options: AIMNET or your own NNP.",
            flush=True,
        )
        logger.info("\tIsomer engine options: RDKit and Omega.")
        logger.info("\tOptimizing engine options: AIMNET or your own NNP.")
        optimizing_engine = args.optimizing_engine
        if optimizing_engine in {"ANI2x", "ANI2xt"}:
            sys.exit(
                f"Only AIMNET can handle: {only_aimnet_smiles}, but {optimizing_engine} was parsed to Auto3D."
            )
            # logger.critical(f"Only AIMNET can handle: {only_aimnet_smiles}, but {optimizing_engine} was parsed to Auto3D.")


def check_smi_format(args):
    ANI_elements = {1, 6, 7, 8, 9, 16, 17}
    ANI = True

    smiles_all = []
    with open(args.path, "r") as f:
        data = f.readlines()
    for line in data:
        if line.isspace():
            continue
        smiles, id = tuple(line.strip().split())
        assert len(smiles) > 0, "Empty SMILES string"
        assert len(id) > 0, "Empty ID"
        # assert "_" not in id, \
        #         f"Sorry, SMILES ID cannot contain underscore: {smiles}"
        # assert "." not in id, \
        #         f"Sorry, SMILES ID cannot contain period: {smiles}"
        smiles_all.append(smiles)
    print(f"\tThere are {len(data)} SMILES in the input file {args.path}. ", flush=True)
    print("\tAll SMILES and IDs are valid.", flush=True)
    logger.info(
        f"\tThere are {len(data)} SMILES in the input file {args.path}. \n\tAll SMILES and IDs are valid."
    )

    # Check number of unspecified atomic stereo center
    if args.enumerate_isomer == False:
        for smiles in smiles_all:
            c = CalcNumUnspecifiedAtomStereoCenters(Chem.MolFromSmiles(smiles))
            if c > 0:
                msg = f"{smiles} contains unspecified atomic stereo centers, but enumerate_isomer=False. Please use enumerate_isomer=True so that Auto3D can enumerate the unspecified atomic stereo centers."
                warnings.warn(msg, UserWarning)

    # Check the properties of molecules
    only_aimnet_smiles = []
    for smiles in smiles_all:
        mol = Chem.MolFromSmiles(smiles)
        charge = Chem.rdmolops.GetFormalCharge(mol)
        elements = set([a.GetAtomicNum() for a in mol.GetAtoms()])
        if (elements.issubset(ANI_elements) is False) or (charge != 0):
            ANI = False
            only_aimnet_smiles.append(smiles)
    return ANI, only_aimnet_smiles


def check_sdf_format(args):
    """
    Check the input file and give recommendations.

    Arguments:
        args: Arguments to auto3d.

    Returns:
        This function checks the format of the input file, the properties for
        each molecule in the input file.
    """
    ANI_elements = {1, 6, 7, 8, 9, 16, 17}
    ANI = True

    supp = Chem.SDMolSupplier(args.path, removeHs=False)
    mols, only_aimnet_ids = [], []
    for mol in supp:
        id = mol.GetProp("_Name")
        assert len(id) > 0, "Empty ID"
        # assert "_" not in id, \
        #         f"Sorry, molecule ID cannot contain underscore: {id}"
        # assert "." not in id, \
        #         f"Sorry, molecule ID cannot contain period: {id}"
        mols.append(mol)

        charge = Chem.rdmolops.GetFormalCharge(mol)
        elements = set([a.GetAtomicNum() for a in mol.GetAtoms()])
        if (elements.issubset(ANI_elements) is False) or (charge != 0):
            ANI = False
            only_aimnet_ids.append(id)
    print(
        f"\tThere are {len(mols)} conformers in the input file {args.path}. ",
        flush=True,
    )
    print("\tAll conformers and IDs are valid.", flush=True)
    logger.info(
        f"\tThere are {len(mols)} conformers in the input file {args.path}. \n\tAll conformers and IDs are valid."
    )

    if args.enumerate_isomer:
        msg = "Enumerating stereocenters of an SDF file could change the conformers of the input file. Please use enumerate_isomer=False."
        warnings.warn(msg, UserWarning)
    return ANI, only_aimnet_ids


def replace_(name, new="-"):
    """
    replace the underscore in a string with user specified character
    name: a string
    new: a tring, lenth=1, used to replace the underscore
    """
    new_name = ""
    for letter in name:
        if letter == "_":
            new_name += new
        else:
            new_name += letter
    return new_name


class NullIO(StringIO):
    """
    Place holder for a clean terminal
    """

    def write(self, txt):
        pass


def hash_enumerated_smi_IDs(smi, out):
    """
    Writes all SMILES with hashed IDs into smiles_enumerated_hashed.smi

    Arguments:
        smi: a .smi File path
        out: the path for the new .smi file where original IDs are hashed.
    Returns:
        writes all SMILES with hashed IDs into smiles_enumerated_hashed.smi
    """
    with open(smi, "r") as f:
        data = f.readlines()

    dict0 = {}
    for line in data:
        smiles, id = line.strip().split()
        while id in dict0.keys():
            id += "_0"
        dict0[id] = smiles

    dict0 = collections.OrderedDict(sorted(dict0.items()))

    # new_smi = out
    # with open(new_smi, 'w+') as f:
    with open(out, "w+") as f:
        for id, smiles in dict0.items():
            molecule = smiles.strip() + " " + id.strip() + "\n"
            f.write(molecule)


def hash_taut_smi(smi, out):
    """
    Writes all SMILES with hashed IDs for tautomers

    Arguments:
        smi: a .smi File path
        out: the path for the new .smi file where original IDs are hashed.
    """
    with open(smi, "r") as f:
        data = f.readlines()

    dict0 = {}
    for line in data:
        smiles, id = line.strip().split()
        c = 1
        id_ = id
        while ("taut" not in id_) or (id_ in dict0.keys()):
            id_ = id + f"@taut{c}"
            c += 1
        dict0[id_] = smiles

    dict0 = collections.OrderedDict(sorted(dict0.items()))

    with open(out, "w+") as f:
        for id, smiles in dict0.items():
            molecule = smiles.strip() + " " + id.strip() + "\n"
            f.write(molecule)


def housekeeping_helper(folder, file):
    basename = os.path.basename(file)
    new_name = os.path.join(folder, basename)
    shutil.move(file, new_name)


def housekeeping(job_name, folder, optimized_structures):
    """
    Moving all meta data into a folder

    Arguments:
        folder: a folder name to contain all meta data
        out: the resulting SDF output
    Returns:
        whe the function is called, it moves all meta data into a folder.
    """
    paths = os.path.join(job_name, "*")
    files = glob.glob(paths)
    for file in files:
        if file != optimized_structures:
            shutil.move(file, folder)

    try:
        paths1 = os.path.join("", "oeomega_*")
        files1 = glob.glob(paths1)
        paths2 = os.path.join("", "flipper_*")
        files2 = glob.glob(paths2)
        files = files1 + files2
        for file in files:
            shutil.move(file, folder)
    except:
        pass


def enantiomer(l1, l2):
    """Check if two lists of stereo centers are enantiomers"""
    indicator = True
    assert len(l1) == len(l2)
    for i in range(len(l1)):
        tp1 = l1[i]
        tp2 = l2[i]
        idx1, stereo1 = tp1
        idx2, stereo2 = tp2
        assert idx1 == idx2
        if stereo1 == stereo2:
            indicator = False
            return indicator
    return indicator


def enantiomer_helper(smiles):
    """get non-enantiomer SMILES from given list of smiles"""
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    stereo_centers = [
        Chem.FindMolChiralCenters(mol, useLegacyImplementation=False) for mol in mols
    ]
    non_enantiomers = []
    non_centers = []
    for i in range(len(stereo_centers)):
        smi = smiles[i]
        stereo = stereo_centers[i]
        indicator = True
        if len(stereo) > 0:
            for stereo_j in non_centers:
                if enantiomer(stereo_j, stereo):
                    indicator = False
                    break
        if indicator:
            non_centers.append(stereo)
            non_enantiomers.append(smi)
    return non_enantiomers


def remove_enantiomers(inpath, out):
    """Removing enantiomers for the input file
    Arguments:
        inpath: input smi
        output: output smi
    """
    with open(inpath, "r") as f:
        data = f.readlines()

    smiles = defaultdict(lambda: [])
    for line in data:
        vals = line.split()
        smi, name = vals[0].strip(), vals[1].strip().split("_")[0].strip()
        smiles[name].append(smi)

    for key, values in smiles.items():
        try:
            new_values = enantiomer_helper(values)
        except:
            new_values = values
            print(f"Enantiomers not removed for {key}", flush=True)
            logger.info(f"Enantiomers not removed for {key}")

        smiles[key] = new_values

    with open(out, "w+") as f:
        for key, val in smiles.items():
            for i in range(len(val)):
                new_key = key + "_" + str(i)
                line = val[i].strip() + " " + new_key + "\n"
                f.write(line)
    return smiles


def check_connectivity(mol: Chem.Mol) -> bool:
    r"""
    Check if there is a new bond formed or a bond broken in the molecule.
    https://github.com/jensengroup/xyz2mol
    """

    block = Chem.MolToXYZBlock(mol, confId=0)
    new_mol = Chem.MolFromXYZBlock(block)
    rdDetermineBonds.DetermineConnectivity(new_mol)
    adjacency_matrix, adjacency_matrix_new = Chem.GetAdjacencyMatrix(
        mol
    ), Chem.GetAdjacencyMatrix(new_mol)
    return np.allclose(adjacency_matrix, adjacency_matrix_new, atol=1e-1)


def is_equal(mol_i: Chem.Mol, mol_j: Chem.Mol, threshold: float) -> bool:
    r"""
    Check if two conformers are the same.
    Caution: This function does not verify whether the two molecules have the same topology!
    """
    # temperoray bug fix for https://github.com/rdkit/rdkit/issues/6826
    # removing Hs speeds up the calculation
    rmsd = rdMolAlign.GetBestRMS(Chem.RemoveHs(mol_i), Chem.RemoveHs(mol_j))
    return rmsd < threshold


def filter_unique(
    mols: Iterable[Chem.Mol], threshold: float = 0.3, k: int = -1
) -> List[Chem.Mol]:
    """
    Remove structures that are very similar.

    Arguments:
        mols: list of rdkit mol objects
        threshold: maximum RMSD for two molecules to be considered the same
        k: only consider the first k unique molecules in the list
    Returns:
        unique_mols: unique molecules
    """

    # Remove similar structures
    unique_mols = []
    for mol_i in mols:
        unique = True
        for mol_j in unique_mols:
            try:
                unique = not is_equal(mol_i=mol_i, mol_j=mol_j, threshold=threshold)
            except RuntimeError:
                unique = False
            if not unique:
                break
        if unique:
            unique_mols.append(mol_i)
        if k > 0 and len(unique_mols) >= k:
            break
    return unique_mols


def no_enantiomer_helper(info1, info2):
    """Return true if info1 and info2 are enantiomers"""
    assert len(info1) == len(info2)
    for i in range(len(info1)):
        if info1[i].strip() == info2[i].strip():
            return False
    return True


def get_stereo_info(smi):
    "Return a dictionary of @@ or @  in smi"
    dct = {}
    regex1 = re.compile("[^@]@[^@]")
    regex2 = re.compile("@@")

    # match @
    for m in regex1.finditer(smi):
        dct[m.start() + 1] = "@"

    # match  @@
    for m in regex2.finditer(smi):
        dct[m.start()] = "@@"

    dct2 = OrderedDict(sorted(dct.items()))
    return dct2


def no_enantiomer(smi, smiles):
    """Return True if there is no enantiomer for smi in smiles"""

    stereo_infoi = list(get_stereo_info(smi).values())
    for i in range(len(smiles)):
        tar = smiles[i]
        if tar != smi:
            stereo_infoj = list(get_stereo_info(tar).values())
            if no_enantiomer_helper(stereo_infoi, stereo_infoj):
                return False
    return True


def create_enantiomer(smi):
    """Create an enantiomer SMILES for input smi"""
    stereo_info = get_stereo_info(smi)
    new_smi = ""
    # for key in stereo_info.keys():
    #     val = stereo_info[key]
    #     if val == '@':
    keys = list(stereo_info.keys())
    if len(keys) == 1:
        key = keys[0]
        val = stereo_info[key]
        if val == "@":
            new_smi += smi[:key]
            new_smi += "@@"
            new_smi += smi[(key + 1) :]
        elif val == "@@":
            new_smi += smi[:key]
            new_smi += "@"
            new_smi += smi[(key + 2) :]
        else:
            raise ValueError("Invalid %s" % smi)
        return new_smi

    for i in range(len(keys)):
        if i == 0:
            key = keys[i]
            new_smi += smi[:key]
        else:
            key1 = keys[i - 1]
            key2 = keys[i]
            val1 = stereo_info[key1]
            if val1 == "@":
                new_smi += "@@"
                new_smi += smi[int(key1 + 1) : key2]
            elif val1 == "@@":
                new_smi += "@"
                new_smi += smi[int(key1 + 2) : key2]
    val2 = stereo_info[key2]
    if val2 == "@":
        new_smi += "@@"
        new_smi += smi[int(key2 + 1) :]
    elif val2 == "@@":
        new_smi += "@"
        new_smi += smi[int(key2 + 2) :]
    return new_smi


def check_value(n):
    """Return True if n is a power of 2. 2^-2 and 2^-1 should be accessible,
    because not all stereo centers can be enumerated. For example: CC12CCC(C1)C(C)(C)C2O"""

    power = math.log(n, 2)
    decimal, integer = math.modf(power)
    i = abs(power - integer)
    if i < 0.0001:
        return True
    return False


def amend_configuration(smis):
    """Adding the missing configurations.
    Example: N=C1OC(CN2CC(C)OC(C)C2)CN1"""

    with open(smis, "r") as f:
        data = f.readlines()
    dct = defaultdict(lambda: [])
    for line in data:
        smi, idx = tuple(line.strip().split())
        idx = idx.split("_")[0].strip()
        dct[idx].append(smi)

    for key in dct.keys():
        value = dct[key]
        smi = value[0]
        mol = Chem.MolFromSmiles(smi)
        num_centers = CalcNumAtomStereoCenters(mol)
        num_unspecified_centers = CalcNumUnspecifiedAtomStereoCenters(mol)
        # num_configurations = 2 ** num_unspecified_centers
        num_configurations = 2**num_centers
        num = len(value) / num_configurations

        if (check_value(num) == False) and ("@" in smi):  # Missing configurations
            try:
                new_value = []
                for val in value:
                    if no_enantiomer(val, value):
                        new_val = create_enantiomer(val)
                        new_value.append(new_val)
                value += new_value

                # assert
                new_num = len(value) / num_configurations
                assert check_value(new_num) == True
                dct[key] = value
            except:
                print(f"Stereo centers for {key} are not fully enumerated.", flush=True)
                logger.info(f"Stereo centers for {key} are not fully enumerated.")
    return dct


def amend_configuration_w(smi):
    """Write the output from dictionary"""
    dct = amend_configuration(smi)
    with open(smi, "w+") as f:
        for key in dct.keys():
            val = dct[key]
            for i, smi in enumerate(val):
                idx = str(key).strip() + "_" + str(i + 1)
                line = smi + " " + idx + "\n"
                f.write(line)


class my_name_space(dict):
    """A modified dictionary whose keys can be accessed via dot. This dict is
    similar to a NameSpace object from parser.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return "<my_name_space " + dict.__repr__(self) + ">"


def min_pairwise_distance(points: np.array) -> float:
    """
    Finds the minimum pairwise distance among the n points provided in a n x 3 matrix.

    Parameters:
    points (numpy.ndarray): A n x 3 matrix representing the coordinates of n points in 3D space.

    Returns:
    float: The minimum pairwise distance among the n points.
    """
    # Ensure input is a NumPy array
    points = points.astype(np.float32)
    n = points.shape[0]
    # Expand dimensions of points to enable broadcasting
    points_expanded = np.expand_dims(points, axis=1).repeat(n, axis=1)

    # Compute pairwise squared differences
    diff_squared = (points_expanded - points_expanded.transpose(1, 0, 2)) ** 2

    # Sum along the last dimension to get pairwise squared distances
    pairwise_squared_distances = np.sum(diff_squared, axis=-1)

    # Find the minimum squared distance
    upp_indices = np.triu_indices(n, 1)
    upp_values = pairwise_squared_distances[upp_indices]
    min_squared_distance = np.min(upp_values)

    # Return the square root of the minimum squared distance
    return np.sqrt(min_squared_distance)


def reorder_sdf(sdf: str, source: str) -> List[Chem.Mol]:
    """Reorder the conformer order in the output SDF file such that
    it's consistent with the order in the input source file"""
    # convert smi/sdf to a list of ids with correct order
    ids = []
    format = guess_file_type(source)
    if format == "smi":
        with open(source, "r") as f:
            data = f.readlines()
        for line in data:
            smiles, id = tuple(line.strip().split())
            ids.append(id)
    elif format == "sdf":
        supp = Chem.SDMolSupplier(source, removeHs=False)
        for mol in supp:
            id = mol.GetProp("_Name")
            ids.append(id)
    else:
        print("Unsupported file format: %s" % format)
        return

    # convert sdf to a Dict[id, List[mols]]
    id_mols = defaultdict(lambda: [])
    supp = Chem.SDMolSupplier(sdf, removeHs=False)
    for mol in supp:
        id = mol.GetProp("_Name")
        if "@taut" in id:
            id = id.split("@taut")[0]
        id_mols[id].append(mol)

    # write the mols in the correct order to a new sdf file
    ordered_mols = []
    with Chem.SDWriter(sdf) as f:
        for id in ids:
            mols = id_mols[id]
            if len(mols) >= 1:
                ordered_mols.extend(mols)
                for mol in mols:
                    f.write(mol)
    return ordered_mols
