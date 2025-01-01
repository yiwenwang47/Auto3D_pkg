# Original source: /labspace/models/aimnet/batch_opt_script/
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.jit import Attribute
from torch.profiler import ProfilerActivity, profile, record_function

try:
    import torchani
except:
    pass

from rdkit import Chem
from rdkit.Chem import rdmolops

try:
    from .ANI2xt_no_rep import ANI2xt
except:
    pass

from tqdm.auto import tqdm

from Auto3D.utils import hartree2ev

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class FIRE(nn.Module):
    r"""
    A general optimization program.

    Implementation based on:
    #Guénolé, Julien, et al. Computational Materials Science 175 (2020): 109584.
    """

    def __init__(
        self, shape: Tuple[int], device: Optional[torch.device] = torch.device("cpu")
    ):
        super(FIRE, self).__init__()
        ## default parameters
        self.dt_max = 0.1
        self.Nmin = 5
        self.maxstep = 0.1
        self.finc = 1.5
        self.fdec = 0.7
        self.astart = 0.1
        self.fa = 0.99
        self.v = Attribute(torch.zeros(*shape, device=device), torch.Tensor)
        self.Nsteps = Attribute(
            torch.zeros(shape[0], dtype=torch.long, device=device), torch.Tensor
        )
        self.dt = Attribute(torch.full(shape[:1], 0.1, device=device), torch.Tensor)
        self.a = Attribute(torch.full(shape[:1], 0.1, device=device), torch.Tensor)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, coord: torch.Tensor, forces: torch.Tensor) -> torch.Tensor:
        """Moving atoms based on forces

        Arguments:
            coord: coordinates of atoms. Size (Batch, N, 3), where Batch is
                   the number of structures, N is the number of atom in each structure.
            forces: forces on each atom. Size (Batch, N, 3).

        Return:
            new coordinates that are moved based on input forces. Size (Batch, N, 3)"""
        vf = (forces * self.v).flatten(-2, -1).sum(-1)
        w_vf = vf > 0.0
        if w_vf.all():
            a = self.a.unsqueeze(-1).unsqueeze(-1)
            v = self.v
            f = forces
            self.v = (1.0 - a) * v + a * v.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(
                -1
            ).unsqueeze(-1) * f / f.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(
                -1
            ).unsqueeze(
                -1
            )
            # self.Nsteps += 1
        elif w_vf.any():
            a = self.a[w_vf].unsqueeze(-1).unsqueeze(-1)
            v = self.v[w_vf]
            f = forces[w_vf]
            self.v[w_vf] = (1.0 - a) * v + a * v.flatten(-2, -1).norm(
                p=2, dim=-1
            ).unsqueeze(-1).unsqueeze(-1) * f / f.flatten(-2, -1).norm(
                p=2, dim=-1
            ).unsqueeze(
                -1
            ).unsqueeze(
                -1
            )

        # Update alpha (a) and delta_t (dt) accordingly
        w_N = self.Nsteps > self.Nmin
        w_vfN = w_vf & w_N
        self.dt[w_vfN] = (self.dt[w_vfN] * self.finc).clamp(
            max=self.dt_max
        )  # Algorithm 2, line 13
        self.a[w_vfN] *= self.fa  # Algorithm 2, line 14

        # Update Nsteps
        self.Nsteps[w_vf] += 1  # Algorithm 2, line 10

        w_vf = ~w_vf
        if w_vf.all():
            self.v[:] = 0.0
            self.a[:] = torch.tensor(self.astart, device=self.a.device)
            self.dt[:] *= self.fdec
            self.Nsteps[:] = 0
        elif w_vf.any():
            self.v[w_vf] = torch.tensor(0.0, device=self.v.device)
            self.a[w_vf] = torch.tensor(self.astart, device=self.a.device)
            self.dt[w_vf] *= self.fdec
            self.Nsteps[w_vf] = torch.tensor(0, device=self.v.device)

        dt = self.dt.unsqueeze(-1).unsqueeze(-1)
        self.v += dt * forces
        dr = dt * self.v
        normdr = dr.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(-1)
        dr *= (self.maxstep / normdr).clamp(max=1.0)
        return coord + dr

    @torch.jit.export
    def clean(self, mask: torch.Tensor):
        self.v = self.v[mask]
        self.Nsteps = self.Nsteps[mask]
        self.dt = self.dt[mask]
        self.a = self.a[mask]


class EnForce_ANI(nn.Module):
    """Takes in an torch model, then defines two forward functions for it.
    The input model should be able to calculate energy and disp_energy given
    coordiantes, species and charges of a molecule.

    Arguments:
        ani: model
        batchsize_atoms: the maximum nmber atoms that can be handled in one batch.

    Returns:
        the energies and forces for the input molecules. One time calculation.
    """

    def __init__(self, ani, name, batchsize_atoms=1024 * 16):
        super().__init__()
        self.add_module("ani", ani)
        self.name = name
        self.batchsize_atoms = batchsize_atoms

    def forward(self, coord, numbers, charges):
        """Calculate the energies and forces for input molecules. Called by self.forward_batched

        Arguments:
            coord: coordinates for all input structures. size (B, N, 3), where
                  B is the number of structures in coord, N is the number of
                  atoms in each structure, 3 represents xyz dimensions.
            numbers: the periodic numbers for all atoms.
            charges: tensor size (B)

        Returns:
            energies
            forces
        """
        if self.name == "AIMNET":
            d = self.ani(
                dict(coord=coord, numbers=numbers, charge=charges)
            )  # Output from the model
            e = d["energy"].to(torch.double)
            f = d["forces"]
        elif self.name == "ANI2xt":
            e = self.ani(numbers, coord)
            g = torch.autograd.grad([e.sum()], [coord])[0]
            f = -g
        elif self.name == "ANI2x":
            e = self.ani((numbers, coord)).energies
            e = (
                e * hartree2ev
            )  # ANI2x (torch.models.ANI2x()) output energy unit is Hatree;
            # ANI ASE interface unit is eV
            g = torch.autograd.grad([e.sum()], [coord])[0]
            f = -g
        else:
            # user NNP that was loaded from a file
            e = self.ani(numbers, coord, charges)
            g = torch.autograd.grad([e.sum()], [coord])[0]
            f = -g

        return e, f

    def forward_batched(self, coord, numbers, charges):
        """Calculate the energies and forces for input molecules.

        Arguments:
            coord: coordinates for all input structures. size (B, N, 3), where
                  B is the number of structures in coord, N is the number of
                  atoms in each structure, 3 represents xyz dimensions.
            numbers: the periodic numbers for all atoms. size (B, N)

        Returns:
            energies
            forces
        """
        B, N = coord.shape[:2]
        e = []
        f = []
        idx = torch.arange(B, device=coord.device)
        for batch in idx.split(self.batchsize_atoms // N):
            _e, _f = self(coord[batch], numbers[batch], charges[batch])
            e.append(_e)
            f.append(_f)
        return torch.cat(e, dim=0), torch.cat(f, dim=0)


def print_stats(state, patience):
    """Print the optimization status"""
    numbers = state["numbers"]
    num_total = numbers.size()[0]
    num_converged_dropped = torch.sum(state["converged_mask"]).to("cpu")
    oscillating_count = (
        state["oscillating_count"]
        .to("cpu")
        .reshape(
            -1,
        )
        >= patience
    )
    num_dropped = torch.sum(oscillating_count)
    num_converged = num_converged_dropped - num_dropped
    num_active = num_total - num_converged_dropped
    print(
        "Total 3D structures: %i  Converged: %i   Dropped(Oscillating): %i    Active: %i"
        % (num_total, num_converged, num_dropped, num_active),
        flush=True,
    )
    # logging.info("Total 3D structures: %i  Converged: %i   Dropped(Oscillating): %i    Active: %i" %
    #       (num_total, num_converged, num_dropped, num_active))


def n_steps(state: dict[torch.Tensor], n: int, opttol: float, patience: int):
    """Doing n steps optimization for each input. Only converged structures are
    modified at each step. n_steps does not change input conformer order.

    Argument:
        state: an dictionary containing all information about this optimization step
        n: optimization step
        patience: optimization stops for a conformer if the force does not decrease for a continuous patience steps"""
    # t0 = perf_counter()
    numbers = state["numbers"]
    charges = state["charges"]
    coord = state["coord"]
    optimizer = FIRE(shape=tuple(coord.shape), device=coord.device)
    optimizer = torch.jit.script(optimizer)
    # the following two terms are used to detect oscillating conformers
    smallest_fmax0 = torch.tensor(np.ones((len(coord), 1)) * 999, dtype=torch.float).to(
        coord.device
    )
    oscillating_count0 = torch.tensor(np.zeros((len(coord), 1)), dtype=torch.float).to(
        coord.device
    )
    state["oscillating_count"] = oscillating_count0
    assert len(coord.shape) == 3
    assert len(numbers.shape) == 2
    assert len(charges.shape) == 1
    assert len(smallest_fmax0.shape) == 2
    assert len(oscillating_count0.shape) == 2
    for istep in tqdm(range(1, (n + 1), 1)):
        not_converged = ~state["converged_mask"]  # Essential tracker handle, size fixed
        # stop optimization if all structures converged.
        if not not_converged.any():
            break

        coord = state["coord"][not_converged]  # Subset coordinates, size=not_converged.
        numbers = state["numbers"][not_converged]
        charges = state["charges"][not_converged]
        smallest_fmax = smallest_fmax0[not_converged]
        oscillating_count = state["oscillating_count"][not_converged]

        coord.requires_grad_(True)
        e, f = state["nn"].forward_batched(
            coord, numbers, charges
        )  # Key step to calculate all energies and forces.
        coord.requires_grad_(False)

        with torch.no_grad():
            coord = optimizer(coord, f)
        fmax = f.norm(dim=-1).max(dim=-1)[
            0
        ]  # Tensor, Norm is the length of each vector. Here it returns the maximum force length for ecah conformer. Size (100)
        assert len(fmax.shape) == 1
        not_converged_post1 = fmax > opttol

        # update smallest_fmax for each molecule
        fmax_reduced = fmax.reshape(-1, 1) < smallest_fmax
        fmax_reduced = fmax_reduced.reshape(
            -1,
        )
        smallest_fmax[fmax_reduced] = fmax.reshape(-1, 1)[fmax_reduced]
        # reduce count to 0 for reducing; raise count for non-reducing
        oscillating_count[fmax_reduced] = 0
        fmax_not_reduced = ~fmax_reduced
        oscillating_count += fmax_not_reduced.reshape(-1, 1)
        not_oscillating = oscillating_count < patience
        not_oscillating = not_oscillating.reshape(
            -1,
        )
        not_converged_post = not_converged_post1 & not_oscillating

        optimizer.clean(not_converged_post)  # Subset v, a in FIRE for next optimization

        state["converged_mask"][
            not_converged
        ] = (
            ~not_converged_post
        )  # Update converged_mask, so that converged structures will not be updated in future steps.
        state["fmax"][
            not_converged
        ] = fmax  # Update fmax for conformers that are optimized in this iteration
        state["energy"][
            not_converged
        ] = (
            e.detach()
        )  # Update energy for conformers that are optimized in this iteration
        state["coord"][
            not_converged
        ] = coord  # Update coordinates for conformers that are optimized in this iteration
        smallest_fmax0[
            not_converged
        ] = smallest_fmax  # update smalles_fmax for each conformer
        state["oscillating_count"][
            not_converged
        ] = oscillating_count  # update counts for continuous no reduction in fmax

        # if (istep % (n // 10)) == 0:
        #     print_stats(state, patience)
    if istep == (n):
        print("Reaching maximum optimization step:   ", end="")
        # logging.info("Reaching maximum optimization step:   ")
    else:
        print(f"Optimization finished at step {istep}:   ", end="")
        # logging.info(f"Optimization finished at step {istep}:   ")
    print_stats(state, patience)


def ensemble_opt(
    net,
    coord: List[List[float]],
    numbers: List[int],
    charges: List[int],
    param: Dict[str, Union[int, float]],
    device: torch.device,
):
    """Optimizing a group of molecules

    Arguments:
    net: an EnForce_ANI object
    coord: coordinates of input molecules (N, m, 3). N is the number of structures
           m is the number of atoms in each structure.
    numbers: atomic numbers in the molecule (include H). (N, m)
    charges: (N,)
    param: a dictionary containing parameters
    model: "AIMNET", "ANI2xt", "ANI2x" or "userNNP"
    device
    """
    coord = torch.tensor(coord, dtype=torch.float, device=device)
    numbers = torch.tensor(numbers, dtype=torch.long, device=device)
    charges = torch.tensor(charges, dtype=torch.long, device=device)
    converged_mask = torch.zeros(coord.shape[0], dtype=torch.bool, device=device)
    fmax = torch.full(
        coord.shape[:1], 999.0, device=coord.device
    )  # size=N, a tensored filled with 999.0, representing the current maximum forces at each conformer.
    energy = torch.full(coord.shape[:1], 999.0, dtype=torch.double, device=coord.device)

    state = dict(
        coord=coord,
        numbers=numbers,
        charges=charges,
        converged_mask=converged_mask,
        nn=net,
        fmax=fmax,
        energy=energy,
        # timing=defaultdict(float),
    )
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True
    ) as prof:
        n_steps(
            state=state,
            # n=param["opt_steps"],
            n=1,
            opttol=param["opttol"],
            patience=param["patience"],
        )
    total_flops = sum(
        entry.flops for entry in prof.key_averages() if hasattr(entry, "flops")
    )
    print(f"Total FLOPs: {total_flops}", flush=True)

    return dict(
        coord=state["coord"].tolist(),
        energy=state["energy"].tolist(),
        fmax=state["fmax"].tolist(),
        # timing=dict(state["timing"]),
    )


def padding_coords(lists, pad_value=0.0):
    lengths = [len(lst) for lst in lists]
    max_length = max(lengths)
    pad_length = [max_length - len(lst) for lst in lists]
    assert len(pad_length) == len(lists)

    lists_padded = []
    for i in range(len(pad_length)):
        lst_i = lists[i]
        pad_i = [(pad_value, pad_value, pad_value) for _ in range(pad_length[i])]
        lst_i_padded = lst_i + pad_i
        lists_padded.append(lst_i_padded)
    return lists_padded


def padding_species(lists, pad_value=-1):
    lengths = [len(lst) for lst in lists]
    max_length = max(lengths)
    pad_length = [max_length - len(lst) for lst in lists]
    assert len(pad_length) == len(lists)

    lists_padded = []
    for i in range(len(pad_length)):
        lst_i = lists[i]
        pad_i = [pad_value for _ in range(pad_length[i])]
        lst_i_padded = lst_i + pad_i
        lists_padded.append(lst_i_padded)
    return lists_padded


def mols2lists(mols, model):
    """mols: rdkit mol object"""
    ani2xt_index = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6}  # H, C, N, O, F, S, Cl
    coord = [mol.GetConformer().GetPositions().tolist() for mol in mols]
    coord = [
        [tuple(xyz) for xyz in inner] for inner in coord
    ]  # to be consistent with legacy code
    charges = [rdmolops.GetFormalCharge(mol) for mol in mols]

    if model == "ANI2xt":
        numbers = [
            [ani2xt_index[a.GetAtomicNum()] for a in mol.GetAtoms()] for mol in mols
        ]
    else:
        numbers = [[a.GetAtomicNum() for a in mol.GetAtoms()] for mol in mols]
    return coord, numbers, charges


class optimizing(object):
    def __init__(self, in_f, out_f, name, device, config):
        self.in_f = in_f
        self.out_f = out_f
        self.name = name
        self.device = device
        self.config = config
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if name == "AIMNET":
            self.model = torch.jit.load(
                os.path.join(root, "models/aimnet2_wb97m_ens_f.jpt"),
                map_location=device,
            )
            self.coord_pad = 0
            self.species_pad = 0
        elif name == "ANI2xt":
            self.model = ANI2xt(device)
            self.coord_pad = 0
            self.species_pad = -1
        elif name == "ANI2x":
            self.model = torchani.models.ANI2x(periodic_table_index=True).to(device)
            self.coord_pad = 0
            self.species_pad = -1
        elif os.path.exists(name):
            print(f"Loading model from {name}", flush=True)
            self.model = torch.jit.load(name, map_location=device)
            self.coord_pad = self.model.coord_pad
            self.species_pad = self.model.species_pad
        else:
            raise ValueError("Model has to be ANI2x, ANI2xt, userNNP or AIMNET.")

    def run(self):
        print(
            "Preparing for parallel optimization... (Max optimization steps: %i)"
            % self.config["opt_steps"],
            flush=True,
        )
        # logging.info("Preparing for parallel optimizing... (Max optimization steps: %i)" % self.config["opt_steps"])
        mols = list(Chem.SDMolSupplier(self.in_f, removeHs=False))
        print(f"Total 3D conformers: {len(mols)}", flush=True)
        # logging.info(f"Total 3D conformers: {len(mols)}")
        coord, numbers, charges = mols2lists(mols, self.name)
        coord_padded = padding_coords(coord, self.coord_pad)
        numbers_padded = padding_species(numbers, self.species_pad)

        for p in self.model.parameters():
            p.requires_grad_(False)
        model = EnForce_ANI(
            self.model, self.name, self.config["batchsize_atoms"]
        )  # Interestingly, EnForce_ANI inherits nn.module, bu can still accept a ScriptModule object as the input

        with torch.jit.optimized_execution(True):
            optdict = ensemble_opt(
                net=model,
                coord=coord_padded,
                numbers=numbers_padded,
                charges=charges,
                param=self.config,
                device=self.device,
            )  # Magic step

        energies = optdict["energy"]
        fmax = optdict["fmax"]
        convergence_mask = list(map(lambda x: (x <= self.config["opttol"]), fmax))

        with Chem.SDWriter(self.out_f) as f:
            for i in range((len(mols))):
                mol = mols[i]
                idx = mol.GetProp("_Name")
                fmax_i = fmax[i]
                mol.SetProp("E_tot", str(energies[i]))
                mol.SetProp("fmax", str(fmax_i))
                mol.SetProp("Converged", str(convergence_mask[i]))
                mol.SetProp("ID", idx)
                coord = optdict["coord"][i]
                for i, atom in enumerate(mol.GetAtoms()):
                    mol.GetConformer().SetAtomPosition(atom.GetIdx(), coord[i])
                f.write(mol)
