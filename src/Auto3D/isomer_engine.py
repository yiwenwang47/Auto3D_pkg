#!/usr/bin/env python
import collections
import glob
import logging
import os
import shutil
import warnings
from typing import Tuple

from rdkit import Chem

Chem.SetUseLegacyStereoPerception(False)
from rdkit.Chem import AllChem, DataStructs, Mol, rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    GetStereoisomerCount,
    StereoEnumerationOptions,
)
from rdkit.Chem.MolStandardize import rdMolStandardize
from send2trash import send2trash

from Auto3D.utils import (
    amend_configuration_w,
    hash_enumerated_smi_IDs,
    hash_taut_smi,
    min_pairwise_distance,
    remove_enantiomers,
)
from Auto3D.utils_file import combine_smi

try:
    from openeye import oechem, oeomega, oequacpac
except:
    pass
from tqdm.auto import tqdm


# logger = logging.getLogger("auto3d")
class tautomer_engine(object):
    """Enemerate possible tautomers for the input_f

    Arguments:
        mode: rdkit or oechem
        input_f: smi file
        output: smi file

    """

    def __init__(self, mode, input_f, out, pKaNorm):
        self.mode = mode
        self.input_f = input_f
        self.output = out
        self.pKaNorm = pKaNorm

    def oe_taut(self):
        """OEChem enumerating tautomers, modified from
        https://docs.eyesopen.com/toolkits/python/quacpactk/examples_summary_getreasonabletautomers.html"""
        ifs = oechem.oemolistream()
        ifs.open(self.input_f)

        ofs = oechem.oemolostream()
        ofs.open(self.output)

        tautomerOptions = oequacpac.OETautomerOptions()

        for mol in ifs.GetOEGraphMols():
            for tautomer in oequacpac.OEGetReasonableTautomers(
                mol, tautomerOptions, self.pKaNorm
            ):
                oechem.OEWriteMolecule(ofs, tautomer)

        # Appending input_f smiles into output
        combine_smi([self.input_f, self.output], self.output)

    def rd_taut(self):
        """RDKit enumerating tautomers"""
        enumerator = rdMolStandardize.TautomerEnumerator()
        smiles = []
        with open(self.input_f, "r") as f:
            data = f.readlines()
            for line in data:
                line = line.strip().split()
                smi, idx = line[0], line[1]
                smiles.append((smi, idx))
        tautomers = []
        for smi_idx in smiles:
            smi, idx = smi_idx
            mol = Chem.MolFromSmiles(smi)
            tauts = enumerator.Enumerate(mol)
            for taut in tauts:
                tautomers.append((Chem.MolToSmiles(taut), idx))
        with open(self.output, "w+") as f:
            for smi_idx in tautomers:
                smi, idx = smi_idx
                line = smi.strip() + " " + str(idx.strip()) + "\n"
                f.write(line)

    def run(self):
        if self.mode == "oechem":
            self.oe_taut()
        elif self.mode == "rdkit":
            self.rd_taut()
        else:
            raise ValueError(f'{self.mode} must be one of "oechem" or "rdkit".')


def to_isomers(mol: Mol) -> list[Mol]:
    r"""
    Recursively enumerate all stereoisomers of a molecule. The official enumerator has trouble with some bicyclic molecules.

    Args:
        mol (Mol): A molecule.

    Returns:
        list[Mol]: A list of stereoisoemrs.
    """
    options = StereoEnumerationOptions(onlyUnassigned=True, unique=True)
    isomers = list(EnumerateStereoisomers(mol, options=options))
    return isomers


class rd_isomer(object):
    """
    Enumerating stereoisomers for each SMILES representation with RDKit.

    Arguments:
        smi: A smi file containing SMILES and IDs.
        smiles_enumerated: A smi containing cis/trans isomers for the smi file.
        smiles_hashed: For smiles_enumerated, each ID is hashed.
        enumerated_sdf: for smiles_hashed, generating possible 3D structures.
        job_name: as the name suggests.
        max_confs: maximum number of conformers for each smi.
        threshold: Maximum RMSD to be considered as duplicates.
    """

    def __init__(
        self,
        smi,
        smiles_enumerated,
        smiles_enumerated_reduced,
        smiles_hashed,
        enumerated_sdf,
        job_name,
        max_confs,
        threshold,
        np,
        flipper=True,
    ):
        self.input_f = smi
        self.n_conformers = max_confs
        self.enumerate = {}
        self.enumerated_smi_path = smiles_enumerated
        self.enumerated_smi_path_reduced = smiles_enumerated_reduced
        self.enumerated_smi_hashed_path = smiles_hashed
        self.enumerated_sdf = enumerated_sdf
        self.num2sym = {1: "H", 6: "C", 8: "O", 7: "N", 9: "F", 16: "S", 17: "Cl"}
        self.rdk_tmp = os.path.join(job_name, "rdk_tmp")
        os.mkdir(self.rdk_tmp)
        self.threshold = threshold
        self.np = np
        self.flipper = flipper

    @staticmethod
    def read(input_f):
        outputs = {}
        with open(input_f, "r") as f:
            data = f.readlines()
        for line in data:
            smiles, name = tuple(line.strip().split())
            outputs[name.strip()] = smiles.strip()
        return outputs

    @staticmethod
    def enumerate_func(mol: Mol) -> list[str]:
        """Enumerate the R/S and cis/trans isomers

        Argument:
            mol: rd mol object

        Return:
            isomers: a list of SMILES"""
        isomers = to_isomers(mol)
        isomer_smiles = sorted(
            Chem.MolToSmiles(x, isomericSmiles=True, doRandom=False) for x in isomers
        )
        return isomer_smiles

    def write_enumerated_smi(self):
        with open(self.enumerated_smi_path, "w+") as f:
            for name, smi in self.enumerate.items():
                for i, isomer in enumerate(smi):
                    new_name = str(name).strip() + "_" + str(i)
                    line = isomer.strip() + "\t" + new_name + "\n"
                    f.write(line)

    def embed_conformer(self, smi: str) -> Chem.Mol:
        """Embed conformers for a smi"""
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        if self.n_conformers is None:
            # The formula is based on this paper: https://doi.org/10.1021/acs.jctc.0c01213
            num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            num_heavy_atoms = mol.GetNumHeavyAtoms()
            n_conformers = min(
                max(num_heavy_atoms, int(2 * 8.481 * (num_rotatable_bonds**1.642))),
                1000,
            )
        else:
            n_conformers = self.n_conformers

        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=n_conformers,
            randomSeed=42,
            numThreads=self.np,
            pruneRmsThresh=self.threshold,
            maxAttempts=10,  # https://github.com/rdkit/rdkit/discussions/6804
        )

        return mol

    def run(self):
        """
        When called, enumerate 3 dimensional structures for the input_f file and
        writes all structures in 'job_name/smiles_enumerated.sdf'
        """
        if self.flipper:
            print(
                "Enumerating cis/tran isomers for unspecified double bonds...",
                flush=True,
            )
            print(
                "Enumerating R/S isomers for unspecified atomic centers...", flush=True
            )
            # logger.info("Enumerating cis/tran isomers for unspecified double bonds...")
            # logger.info("Enumerating R/S isomers for unspecified atomic centers...")
            smiles_og = self.read(self.input_f)
            for name, smiles in smiles_og.items():
                # mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                mol = Chem.MolFromSmiles(smiles)
                isomers = self.enumerate_func(mol)
                self.enumerate[name] = isomers
            self.write_enumerated_smi()
            print("Removing enantiomers...", flush=True)
            # logger.info("Removing enantiomers...")
            amend_configuration_w(self.enumerated_smi_path)
            remove_enantiomers(
                self.enumerated_smi_path, self.enumerated_smi_path_reduced
            )
            hash_enumerated_smi_IDs(
                self.enumerated_smi_path_reduced, self.enumerated_smi_hashed_path
            )
        else:
            hash_enumerated_smi_IDs(self.input_f, self.enumerated_smi_hashed_path)

        print("Enumerating conformers/rotamers, removing duplicates...", flush=True)
        # logger.info("Enumerating conformers/rotamers, removing duplicates...")
        smiles2 = self.read(self.enumerated_smi_hashed_path)

        smi_name_tuples = [(smi, name) for name, smi in smiles2.items()]

        with Chem.SDWriter(self.enumerated_sdf) as writer:
            for smi, name in tqdm(smi_name_tuples):
                mol = self.embed_conformer(smi)
                for i in range(mol.GetNumConformers()):
                    positions = mol.GetConformer(i).GetPositions()
                    # atoms clash if min distance is smaller than 0.9 Angstrom
                    if min_pairwise_distance(positions) < 0.9:
                        AllChem.MMFFOptimizeMolecule(mol, confId=i)
                    positions = mol.GetConformer(i).GetPositions()
                    if min_pairwise_distance(positions) > 0.9:
                        conf_id = name.strip() + f"_{i}"
                        mol.SetProp("ID", conf_id)
                        mol.SetProp("_Name", conf_id)
                        writer.write(mol, confId=i)

        return self.enumerated_sdf


class rd_isomer_sdf(object):
    """
    enumerating conformers starting from an SDF file.
    The specified stereo centers are preserved as in the input_f file.
    The unspecified stereo centers are enumerated.
    """

    def __init__(
        self, sdf: str, enumerated_sdf: str, max_confs: int, threshold: float, np: int
    ):
        """
        sdf: the path to the input_f sdf file
        enumerated_sdf: the path to the output sdf file
        max_confs: the maximum number of conformers to be enumerated for each molecule
        threshold: the RMSD threshold for removing duplicate conformers for each molecule
        np: the number of threads to be used for parallelization
        """
        self.sdf = sdf
        self.enumerated_sdf = enumerated_sdf
        self.n_conformers = max_confs
        self.threshold = threshold
        self.np = np

    def run(self):
        supp = Chem.SDMolSupplier(self.sdf, removeHs=False)
        with Chem.SDWriter(self.enumerated_sdf) as writer:
            for mol in tqdm(supp):
                # enumerate conformers
                mol2 = Chem.AddHs(mol)
                if self.n_conformers is None:
                    # n_conformers = min(3 ** num_rotatable_bonds, 100)

                    # The formula is based on this paper: https://doi.org/10.1021/acs.jctc.0c01213
                    num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
                    num_heavy_atoms = len(
                        [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
                    )
                    n_conformers = min(
                        max(
                            num_heavy_atoms,
                            int(2 * 8.481 * (num_rotatable_bonds**1.642)),
                        ),
                        1000,
                    )
                else:
                    n_conformers = self.n_conformers
                AllChem.EmbedMultipleConfs(
                    mol2,
                    numConfs=n_conformers,
                    randomSeed=42,
                    numThreads=self.np,
                    pruneRmsThresh=self.threshold,
                    maxAttempts=10,
                )
                # set conformer names
                name = mol.GetProp("_Name")
                for i, conf in enumerate(mol2.GetConformers()):
                    # mol2.ClearProp('ID')
                    # mol2.ClearProp('_Name')
                    mol2.SetProp("_Name", f"{name}_{i}")
                    mol2.SetProp("ID", f"{name}_{i}")
                    writer.write(mol2, confId=i)
        return self.enumerated_sdf


def oe_flipper(input_f, out):
    """helper function for oe_isomer"""
    ifs = oechem.oemolistream()
    ifs.open(input_f)
    ofs = oechem.oemolostream()
    ofs.open(out)

    flipperOpts = oeomega.OEFlipperOptions()
    flipperOpts.SetWarts(True)
    flipperOpts.SetMaxCenters(12)
    flipperOpts.SetEnumNitrogen(True)
    flipperOpts.SetEnumBridgehead(True)
    flipperOpts.SetEnumEZ(False)
    flipperOpts.SetEnumRS(False)
    for mol in ifs.GetOEMols():
        for enantiomer in oeomega.OEFlipper(mol.GetActive(), flipperOpts):
            enantiomer = oechem.OEMol(enantiomer)
            oechem.OEWriteMolecule(ofs, enantiomer)


def oe_isomer(
    mode,
    input_f,
    smiles_enumerated,
    smiles_reduced,
    output,
    max_confs,
    threshold,
    flipper=True,
):
    """Generating R/S, cis/trans and conformers using omega
    Arguments:
        mode: 'classic', 'macrocycle', 'dense', 'pose', 'rocs' or 'fast_rocs'
        input_f: input_f smi file
        output: output SDF file
        flipper: optional R/S and cis/trans enumeration"""
    input_format = os.path.basename(input_f).split(".")[-1].strip()
    if max_confs is None:
        max_confs = 1000
    if mode == "classic":
        omegaOpts = oeomega.OEOmegaOptions()
    elif mode == "dense":
        omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Dense)
    elif mode == "pose":
        omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Pose)
    elif mode == "rocs":
        omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_ROCS)
    elif mode == "fast_rocs":
        omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_FastROCS)
    elif mode == "macrocycle":
        omegaOpts = oeomega.OEMacrocycleOmegaOptions()
    else:
        raise ValueError(
            f"mode has to be 'classic' or 'macrocycle', but received {mode}."
        )
    omegaOpts.SetParameterVisibility(oechem.OEParamVisibility_Hidden)
    omegaOpts.SetParameterVisibility("-rms", oechem.OEParamVisibility_Simple)
    omegaOpts.SetParameterVisibility("-ewindow", oechem.OEParamVisibility_Simple)
    omegaOpts.SetParameterVisibility("-maxconfs", oechem.OEParamVisibility_Simple)

    if mode == "macrocycle":
        omegaOpts.SetIterCycleSize(1000)
        omegaOpts.SetMaxIter(2000)
        omegaOpts.SetMaxConfs(max_confs)
        omegaOpts.SetEnergyWindow(999)
    else:
        omegaOpts.SetFixRMS(
            threshold
        )  # macrocycle mode does not have the attribute 'SetFixRMS'
        omegaOpts.SetStrictStereo(False)
        omegaOpts.SetWarts(True)
        omegaOpts.SetMaxConfs(max_confs)
        omegaOpts.SetEnergyWindow(999)
        omegaOpts.SetRMSRange("0.8, 1.0, 1.2, 1.4")
    # dense, pose, rocs, fast_rocs mdoes use the default parameters from OEOMEGA:
    # https://docs.eyesopen.com/toolkits/python/omegatk/OEConfGenConstants/OEOmegaSampling.html
    opts = oechem.OESimpleAppOptions(
        omegaOpts, "Omega", oechem.OEFileStringType_Mol, oechem.OEFileStringType_Mol3D
    )

    omegaOpts.UpdateValues(opts)
    if mode == "macrocycle":
        omega = oeomega.OEMacrocycleOmega(omegaOpts)
    else:
        omega = oeomega.OEOmega(omegaOpts)
    if input_format == "smi":
        if flipper:
            print("Enumerating stereoisomers.", flush=True)
            # logger.info("Enumerating stereoisomers.")
            oe_flipper(input_f, smiles_enumerated)
            amend_configuration_w(smiles_enumerated)
            remove_enantiomers(smiles_enumerated, smiles_reduced)
            ifs = oechem.oemolistream()
            ifs.open(smiles_reduced)
        else:
            ifs = oechem.oemolistream()
            ifs.open(input_f)
    elif input_format == "sdf":
        ifs = oechem.oemolistream()
        ifs.open(input_f)
    ofs = oechem.oemolostream()
    ofs.open(output)

    print("Enumerating conformers.", flush=True)
    # logger.info("Enumerating conformers.")
    for mol in tqdm(ifs.GetOEMols()):
        ret_code = omega.Build(mol)
        if ret_code == oeomega.OEOmegaReturnCode_Success:
            oechem.OEWriteMolecule(ofs, mol)
        else:
            oechem.OEThrow.Warning(
                "%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code))
            )

    return 0


def handle_tautomers(path, meta, config, logger):
    output_taut = meta["output_taut"]
    taut_mode = config.tauto_engine
    print("Enumerating tautomers for the input...", end="", flush=True)
    logger.info("Enumerating tautomers for the input...")
    taut_engine = tautomer_engine(taut_mode, path, output_taut, config.pKaNorm)
    taut_engine.run()
    hash_taut_smi(output_taut, output_taut)
    print(f"Tautomers are saved in {output_taut}", flush=True)
    logger.info(f"Tautomers are saved in {output_taut}")
    return output_taut


def generate_isomers_as_sdf(path, directory, meta, config):
    smiles_enumerated = meta["smiles_enumerated"]
    smiles_reduced = meta["smiles_reduced"]
    smiles_hashed = meta["smiles_hashed"]
    enumerated_sdf = meta["enumerated_sdf"]
    max_confs = config.max_confs
    duplicate_threshold = config.threshold
    mpi_np = config.mpi_np
    enumerate_isomer = config.enumerate_isomer
    isomer_program = config.isomer_engine
    # Isomer enumeration step
    if isomer_program == "omega":
        mode_oe = config.mode_oe
        oe_isomer(
            mode=mode_oe,
            input_f=path,
            smiles_enumerated=smiles_enumerated,
            smiles_reduced=smiles_reduced,
            output=enumerated_sdf,
            max_confs=max_confs,
            threshold=duplicate_threshold,
            flipper=enumerate_isomer,
        )
    elif isomer_program == "rdkit":
        if config.input_format == "smi":
            engine = rd_isomer(
                smi=path,
                smiles_enumerated=smiles_enumerated,
                smiles_enumerated_reduced=smiles_reduced,
                smiles_hashed=smiles_hashed,
                enumerated_sdf=enumerated_sdf,
                job_name=directory,
                max_confs=max_confs,
                threshold=duplicate_threshold,
                np=mpi_np,
                flipper=enumerate_isomer,
            )
        elif config.input_format == "sdf":
            engine = rd_isomer_sdf(
                sdf=path,
                enumerated_sdf=enumerated_sdf,
                max_confs=max_confs,
                threshold=duplicate_threshold,
                np=mpi_np,
            )
        engine.run()
    else:
        raise ValueError(
            'The isomer enumeration engine must be "omega" or "rdkit", '
            f"but {config.isomer_engine} was parsed. "
            "You can set the parameter by appending the following:"
            "--isomer_engine=rdkit"
        )
    return enumerated_sdf
