import argparse
import sys

import yaml

import Auto3D
from Auto3D.auto3D import generate_and_optimize_conformers


def int_or_intlist(string):
    try:
        # Try to convert the entire string to an integer
        return int(string)
    except ValueError:
        # If it fails, assume it's a comma-separated list of integers
        return [int(item) for item in string.split(",")]


def cli():
    if len(sys.argv) == 2:
        # using yaml input
        args_dict = yaml.load(open(sys.argv[1], "r"), Loader=yaml.FullLoader)
        # change 'None' to None
        for key, val in args_dict.items():
            if val == "None":
                args_dict[key] = None

    else:
        # using argparse
        parser = argparse.ArgumentParser(
            prog="Auto3D",
            description="Automatic generation of the low-energy 3D structures from ANI neural network potentials",
        )

        parser.add_argument(
            "path", type=str, help="a path of smi/SDF file to store all SMILES and IDs"
        )
        parser.add_argument(
            "--k",
            type=int,
            default=None,
            help="Outputs the top-k structures for each SMILES.",
        )
        parser.add_argument(
            "--window",
            type=float,
            default=None,
            help=(
                "Outputs the structures whose energies are within "
                "window (kcal/mol) from the lowest energy"
            ),
        )
        parser.add_argument(
            "--memory",
            type=int,
            default=None,
            help="The RAM size assigned to Auto3D (unit GB)",
        )
        parser.add_argument(
            "--capacity",
            type=int,
            default=40,
            help="This is the number of SMILES that each 1 GB of memory can handle",
        )
        parser.add_argument(
            "--enumerate_tautomer",
            default=False,
            type=lambda x: (str(x).lower() == "true"),
            help="When True, enumerate tautomers for the input",
        )
        parser.add_argument(
            "--tauto_engine",
            type=str,
            default="rdkit",
            help="Programs to enumerate tautomers, either 'rdkit' or 'oechem'",
        )
        parser.add_argument(
            "--pKaNorm",
            default=True,
            type=lambda x: (str(x).lower() == "true"),
            help="When True, the ionization state of each tautomer will be assigned to a predominant state at ~7.4 (Only works when tauto_engine='oechem')",
        )
        parser.add_argument(
            "--isomer_engine",
            type=str,
            default="rdkit",
            help=(
                "The program for generating 3D isomers for each "
                "SMILES. This parameter is either "
                "rdkit or omega"
            ),
        )
        parser.add_argument(
            "--max_confs",
            type=int,
            default=None,
            help=(
                "Maximum number of isomers for each configuration of the SMILES.",
                "Default is None, and Auto3D will uses a dynamic conformer number for each SMILES.",
            ),
        )
        parser.add_argument(
            "--enumerate_isomer",
            default=True,
            type=lambda x: (str(x).lower() == "true"),
            help="When True, unspecified cis/trans and r/s isomers are enumerated.",
        )
        parser.add_argument(
            "--mode_oe",
            type=str,
            default="classic",
            help=(
                "The mode that omega program will take. It can be either 'classic', 'macrocycle', 'dense', 'pose', 'rocs' or 'fast_rocs'. By default, the 'classic' mode is used."
            ),
        )
        parser.add_argument(
            "--mpi_np",
            type=int,
            default=4,
            help="Number of CPU cores for the isomer generation step.",
        )
        parser.add_argument(
            "--optimizing_engine",
            type=str,
            default="AIMNET",
            help=(
                "Choose either 'ANI2x', 'ANI2xt' or 'AIMNET' for energy "
                "calculation and geometry optimization."
            ),
        )
        parser.add_argument(
            "--use_gpu",
            default=True,
            type=lambda x: (str(x).lower() == "true"),
            help="If True, the program will use GPUs.",
        )
        parser.add_argument(
            "--gpu_idx",
            default=0,
            type=int_or_intlist,
            help="GPU index or indices as a single value or comma-separated list (e.g., 0,1,2)",
        )
        parser.add_argument(
            "--opt_steps",
            type=int,
            default=5000,
            help="Maximum optimization steps for each structure.",
        )
        parser.add_argument(
            "--convergence_threshold",
            type=float,
            default=0.003,
            help="Optimization is considered as converged if maximum force is below this threshold. Unit eV/Angstrom.",
        )
        parser.add_argument(
            "--patience",
            type=int,
            default=1000,
            help="If the force does not decrease for a continuous patience steps, the conformer will be dropped out of the optimization loop.",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.3,
            help=(
                "If the RMSD between two conformers are within threhold, "
                "they are considered as duplicates. One of them will be removed."
            ),
        )
        parser.add_argument(
            "--verbose",
            default=False,
            type=lambda x: (str(x).lower() == "true"),
            help="When True, save all meta data while running.",
        )
        parser.add_argument(
            "--job_name",
            default="",
            type=str,
            help="A folder that stores all the results. By default, the name is the current date and time.",
        )

        args = parser.parse_args()
        args_dict = vars(args)

    print(
        rf"""
         _              _             _____   ____
        / \     _   _  | |_    ___   |___ /  |  _ \
       / _ \   | | | | | __|  / _ \    |_ \  | | | |
      / ___ \  | |_| | | |_  | (_) |  ___) | | |_| |
     /_/   \_\  \__,_|  \__|  \___/  |____/  |____/  {str(Auto3D.__version__)}
        // Generating low-energy 3D structures
    """
    )
    generate_and_optimize_conformers(**args_dict)


if __name__ == "__main__":
    cli()
