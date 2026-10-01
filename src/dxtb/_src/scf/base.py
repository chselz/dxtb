# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Self-consistent field
=====================
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import torch
from tad_mctc.batch import real_atoms
from tad_mctc.math import einsum
from tad_mctc.units import KELVIN2AU

from dxtb import IndexHelper, OutputHandler
from dxtb._src.components.interactions.container import (
    Charges,
    ContainerLayout,
    Potential,
)
from dxtb._src.constants import defaults, labels
from dxtb._src.timing.decorator import timer_decorator
from dxtb._src.typing import (
    DD,
    Any,
    Callable,
    Literal,
    Slicers,
    Tensor,
    overload,
)
from dxtb._src.wavefunction import filling, mulliken, spin
from dxtb.config import ConfigSCF

from .result import SCFResult
from .utils import get_density, resolve_nspin

if TYPE_CHECKING:
    from dxtb._src.components.interactions import (
        InteractionList,
        InteractionListCache,
    )
    from dxtb._src.exlibs import xitorch as xt
    from dxtb._src.integral.container import IntegralMatrices
del TYPE_CHECKING

__all__ = ["BaseSCF"]


class BaseSCF:
    """
    Self-consistent field iterator, which can be used to obtain a
    self-consistent solution for a given Hamiltonian.

    This base class lacks the `scf` method, which implements mixing and
    convergence. Additionally, the `get_overlap` and `diagonalize` method
    must be implemented.
    """

    class _Data:
        """
        Restart data for the singlepoint calculation.
        """

        numbers: Tensor
        """Atomic numbers"""

        integrals: IntegralMatrices
        """
        Collection of integrals. Core Hamiltonian and overlap are always needed.
        """

        occupation: Tensor
        """Occupation numbers (shape: [..., 2, orbs])"""

        n0: Tensor
        """Reference occupation for each orbital (shape: [..., orbs])"""

        ihelp: IndexHelper
        """Index mapping for the basis set"""

        cache: InteractionListCache
        """Restart data for the interactions"""

        energy: Tensor
        """Electronic energy (shape: [..., orbs])"""

        hamiltonian: Tensor
        """Self-consistent Hamiltonian (shape: [..., orbs, orbs])"""

        density: Tensor
        """Density matrix"""

        evals: Tensor
        """
        Orbital energies, i.e., eigenvalues of Fock matrix
        (shape: [..., orbs])
        """

        evecs: Tensor
        """
        LCAO coefficients, i.e., eigenvectors of Fock matrix
        (shape: [..., orbs, orbs])
        """

        iter: int
        """Number of iterations."""

        def __init__(
            self,
            occupation: Tensor,
            n0: Tensor,
            numbers: Tensor,
            ihelp: IndexHelper,
            cache: InteractionListCache,
            integrals: IntegralMatrices,
            nspin: int = 1,
        ) -> None:
            if integrals.hcore is None:
                raise ValueError("No core Hamiltonian provided.")
            if integrals.overlap is None:
                raise ValueError("No Overlap provided.")

            self.ints = integrals
            self.occupation = occupation
            self.n0 = n0
            self.numbers = numbers
            self.ihelp = ihelp
            self.cache = cache
            self.nspin = nspin
            self.init_zeros()

            self.potential = ContainerLayout(nspin=nspin)
            self.charges = ContainerLayout(nspin=nspin)

            # bumped in SCF function (start: 1), guess energy NOT printed (0)
            self.iter = 0

        def init_zeros(self) -> None:
            """Initialize all tensors with zeros."""
            self.energy = torch.zeros_like(self.n0)

            hamiltonian = torch.zeros_like(self.ints.hcore)
            evals = torch.zeros_like(self.n0)
            if self.nspin == 2:
                hamiltonian = torch.stack((hamiltonian, hamiltonian), dim=-3)
                evals = torch.stack((evals, evals), dim=-2)

            self.hamiltonian = hamiltonian
            self.density = torch.zeros_like(hamiltonian)
            self.evals = evals
            self.evecs = torch.zeros_like(hamiltonian)

            self.old_charges = (
                torch.zeros_like(self.energy)
                if self.nspin == 1
                else torch.zeros_like(self.evals)
            )
            self.old_energy = torch.zeros_like(self.numbers)
            self.old_density = torch.zeros_like(self.density)

        def reset(self) -> None:
            """Reset all tensors and iteration count to zero."""
            self.iter = 1
            self.init_zeros()

        def cull(self, conv: Tensor, slicers: Slicers) -> None:
            """
            Cull all tensors according to the given slicers.

            Parameters
            ----------
            conv : Tensor
                Convergence mask.
            slicers : Slicers
                Slicers for the tensors.
            """
            onedim = tuple([~conv, *slicers["orbital"]])
            spin_onedim = tuple([~conv, (...), *slicers["orbital"]])
            onedim_atom = tuple([~conv, *slicers["atom"]])
            twodim = tuple([~conv, *slicers["orbital"], *slicers["orbital"]])
            threedim = tuple(
                [~conv, (...), *slicers["orbital"], *slicers["orbital"]]
            )

            # disable shape check temporarily for writing culled versions back
            self.ints.run_checks = False
            self.ints.overlap = self.ints.overlap[twodim]
            self.ints.hcore = self.ints.hcore[twodim]
            if self.ints.dipole is not None:
                self.ints.dipole = self.ints.dipole[threedim]
            if self.ints.quadrupole is not None:
                self.ints.quadrupole = self.ints.quadrupole[threedim]
            self.ints.run_checks = True

            self.numbers = self.numbers[onedim_atom]
            self.hamiltonian = self.hamiltonian[threedim]
            self.density = self.density[threedim]
            self.occupation = self.occupation[spin_onedim]
            self.evecs = self.evecs[threedim]
            self.evals = self.evals[spin_onedim]
            self.energy = self.energy[onedim]
            self.n0 = self.n0[onedim]
            self.ihelp.cull(conv, slicers=slicers)
            self.cache.cull(conv, slicers=slicers)

            self.old_charges = self.old_charges[spin_onedim]
            self.old_energy = self.old_energy[onedim_atom]
            self.old_density = self.old_density[threedim]

    _data: _Data
    """Persistent data"""

    config: ConfigSCF
    """Configuration object for the SCF procedure."""

    interactions: InteractionList
    """Interactions to minimize in self-consistent iterations"""

    fwd_options: dict[str, Any]
    """Options for forwards pass"""

    bck_options: dict[str, Any]
    """Options for backwards pass"""

    eigen_options: dict[str, Any]
    """Options for eigensolver"""

    batch_mode: int
    """Whether multiple systems or a single one are handled"""

    def __init__(
        self,
        interactions: InteractionList,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if "config" in kwargs:
            self.config = kwargs.pop("config")
            if not isinstance(self.config, ConfigSCF):
                raise ValueError("Invalid configuration object.")
        else:
            self.config = ConfigSCF()

        # TODO: Move these settings to config
        self.bck_options = {"posdef": True, **kwargs.pop("bck_options", {})}
        self.fwd_options = {
            "force_convergence": False,
            "method": "broyden1",
            "alpha": -0.5,
            "damp": self.config.damp,
            "damp_init": self.config.damp_init,
            "damp_dynamic": self.config.damp_dynamic,
            "damp_dynamic_factor": self.config.damp_dynamic_factor,
            "damp_generations": self.config.damp_generations,
            "damp_soft_start": self.config.damp_soft_start,
            "f_tol": self.config.f_atol,
            "x_tol": self.config.x_atol,
            "x_tol_max": self.config.x_atol_max,
            "f_rtol": float("inf"),
            "x_rtol": float("inf"),
            "maxiter": self.config.maxiter,
            "verbose": False,
            "line_search": False,
            **kwargs.pop("fwd_options", {}),
        }

        self.eigen_options = {
            "method": "exacteig",
            **kwargs.pop("eigen_options", {}),
        }

        self._fcn: Callable[[Tensor], Tensor]
        if self.config.scp_mode == labels.SCP_MODE_CHARGE:
            self._fcn = self.iterate_charges
        elif self.config.scp_mode == labels.SCP_MODE_POTENTIAL:
            self._fcn = self.iterate_potential
        elif self.config.scp_mode == labels.SCP_MODE_FOCK:
            self._fcn = self.iterate_fockian
        else:
            raise ValueError(
                f"Unknown convergence target (SCP mode) '{self.config.scp_mode}'."
            )

        self.interactions = interactions
        self.nspin = resolve_nspin(interactions, kwargs.pop("nspin", 1))

        data_kwargs = {**kwargs, "nspin": self.nspin}
        self._data = self._Data(*args, **data_kwargs)

        self.kt = torch.tensor(self.config.fermi.etemp * KELVIN2AU, **self.dd)

    @overload
    @abstractmethod
    def scf(
        self,
        guess: Tensor,
        return_charges: Literal[True] = True,
    ) -> Charges: ...

    @overload
    @abstractmethod
    def scf(
        self,
        guess: Tensor,
        return_charges: Literal[False] = False,
    ) -> Charges | Potential | Tensor: ...

    @abstractmethod
    def scf(
        self, guess: Tensor, return_charges: bool = True
    ) -> Charges | Potential | Tensor:
        """
        Mixing and convergence for self-consistent field iterations.

        Parameters
        ----------
        guess : Tensor
            Orbital-resolved guess for charges, potential or Fock matrix.
        return_charges : bool, optional
            Whether to return the charges. Default is ``True``.

        Returns
        -------
        Tensor
            Converged, orbital-resolved charges.
        """

    @abstractmethod
    def get_overlap(self) -> xt.LinearOperator | Tensor:
        """
        Get the overlap matrix.

        Returns
        -------
        LinearOperator | Tensor
            Overlap matrix.
        """

    @abstractmethod
    def diagonalize(self, hamiltonian: Tensor) -> tuple[Tensor, Tensor]:
        """
        Diagonalize the Hamiltonian.

        The overlap matrix is retrieved within this method using the
        `get_overlap` method.

        Parameters
        ----------
        hamiltonian : Tensor
            Current Hamiltonian matrix.

        Returns
        -------
        evals : Tensor
            Eigenvalues of the Hamiltonian.
        evecs : Tensor
            Eigenvectors of the Hamiltonian.
        """

    def _guess(self, charges: Tensor | Charges | None) -> Tensor:
        """
        Get the initial guess for the charges depending on the convergence
        property (self-consistent property).

        Parameters
        ----------
        charges : Tensor | Charges, optional
            Initial orbital charges vector. If ``None`` is given (default), a
            zero vector is used.

        Returns
        -------
        Tensor
            Initial guess for the charges.
        """

        # initialize zero charges (equivalent to SAD guess)
        if charges is None:
            charges = torch.zeros_like(self._data.n0)

        # initialize Charge container depending on given integrals
        if isinstance(charges, Tensor):
            # guess for total charges; start without magnetization
            if self.nspin == 2:
                charges = torch.stack(
                    (charges, torch.zeros_like(charges)), dim=-2
                )
            charges = Charges(
                mono=charges,
                batch_mode=self.config.batch_mode,
                nspin=self.nspin,
            )

        # restricted guess: start without magnetization
        elif charges.nspin != self.nspin:
            charges = charges.to_spin_channels(0)

        spin_shape = (self.nspin,) if self.nspin == 2 else ()
        atom_shape = (
            *self._data.numbers.shape[:-1],
            *spin_shape,
            self._data.numbers.shape[-1],
        )
        if self._data.ints.dipole is not None and charges.dipole is None:
            charges.dipole = torch.zeros(
                (*atom_shape, defaults.DP_SHAPE), **self.dd
            )

        if self._data.ints.quadrupole is not None and charges.quad is None:
            charges.quad = torch.zeros(
                (*atom_shape, defaults.QP_SHAPE), **self.dd
            )

        self._data.charges = charges.layout

        if self.config.scp_mode == labels.SCP_MODE_CHARGE:
            return charges.as_tensor()

        if self.config.scp_mode == labels.SCP_MODE_POTENTIAL:
            potential = self.charges_to_potential(charges)
            return potential.as_tensor()

        if self.config.scp_mode == labels.SCP_MODE_FOCK:
            potential = self.charges_to_potential(charges)
            return self.potential_to_hamiltonian(potential)

        lbls = (
            labels.SCP_MODE_POTENTIAL_STRS
            + labels.SCP_MODE_CHARGE_STRS
            + labels.SCP_MODE_FOCK_STRS
        )
        raise ValueError(
            "Unknown convergence target (SCP mode) '"
            f"{self.config.scp_mode}'. Use one of {', '.join(lbls)}."
        )

    def __call__(self, charges: Tensor | Charges | None = None) -> SCFResult:
        """
        Run the self-consistent iterations until a stationary solution is
        reached.

        Parameters
        ----------
        charges : Tensor | Charges, optional
            Initial orbital charges vector. If ``None`` is given (default), a
            zero vector is used.

        Returns
        -------
        Tensor
            Converged orbital charges vector.
        """

        guess = self._guess(charges)

        # main SCF function (mixing)
        OutputHandler.write_stdout(
            f"\n{'iter':<5} {'Energy':<24} {'Delta E':<16}"
            f"{'Delta Pnorm':<15} {'Delta q':<15}",
            v=3,
        )
        OutputHandler.write_stdout(77 * "-", v=3)

        q = self.scf(guess)
        q.nullify_padding()

        # evaluate final energy
        energy = self.get_energy(q)
        fenergy = self.get_electronic_free_energy()

        OutputHandler.write_stdout(77 * "-", v=3)
        OutputHandler.write_stdout("", v=3)

        return {
            "charges": q,
            "coefficients": self._data.evecs,
            "density": self._data.density,
            "emo": self._data.evals,
            "energy": energy,
            "fenergy": fenergy,
            "hamiltonian": self._data.hamiltonian,
            "nspin": self.nspin,
            "occupation": self._data.occupation,
            "potential": self.charges_to_potential(q),
            "iterations": self._data.iter,
        }

    def converged_to_charges(self, x: Tensor) -> Charges:
        """
        Convert the converged property to charges.

        Parameters
        ----------
        x : Tensor
            Converged property (scp).

        Returns
        -------
        Tensor
            Orbital-resolved partial charges

        Raises
        ------
        ValueError
            Unknown `scp_mode` given.
        """

        if self.config.scp_mode == labels.SCP_MODE_CHARGE:
            return Charges.from_tensor(
                x, self._data.charges, batch_mode=self.config.batch_mode
            )

        if self.config.scp_mode == labels.SCP_MODE_POTENTIAL:
            pot = Potential.from_tensor(
                x, self._data.potential, batch_mode=self.config.batch_mode
            )
            return self.potential_to_charges(pot)

        if self.config.scp_mode == labels.SCP_MODE_FOCK:
            zero = torch.tensor(0.0, **self.dd)
            x = torch.where(x != defaults.PADNZ, x, zero)

            self._data.density = self.hamiltonian_to_density(x)
            return self.density_to_charges(self._data.density)

        raise ValueError(
            f"Unknown convergence target (SCP mode) '{self.config.scp_mode}'."
        )

    def get_energy(self, charges: Charges) -> Tensor:
        """
        Get the energy of the system with the given charges.

        Parameters
        ----------
        charges : Tensor
            Orbital charges vector (shape: ``(..., nao)``).

        Returns
        -------
        Tensor
            Energy of the system.
        """

        e_h0 = self._data.ihelp.reduce_orbital_to_atom(self._data.energy)
        e_h1 = self.interactions.get_energy(
            charges, self._data.cache, self._data.ihelp
        )

        return e_h0 + e_h1

    def get_energy_as_dict(self, charges: Charges) -> dict[str, Tensor]:
        """
        Get the energy of the system with the given charges.

        Parameters
        ----------
        charges : Tensor
            Orbital charges vector (shape: ``(..., nao)``).

        Returns
        -------
        Tensor
            Energy of the system.
        """
        energy_h0 = {"h0": self._data.energy}

        energy_interactions = self.interactions.get_energy_as_dict(
            charges, self._data.cache, self._data.ihelp
        )

        return {**energy_h0, **energy_interactions}

    def get_electronic_free_energy(self) -> Tensor:
        r"""
        Calculate electronic free energy from entropy.

        .. math::

            G = -TS = k_B\sum_{i}f_i \; ln(f_i) + (1 - f_i)\; ln(1 - f_i))

        The atomic partitioning can be performed by means of Mulliken population
        analysis using an "electronic entropy" density matrix.

        .. math::

            E_\kappa^\text{TS} = (\mathbf P^\text{TS} \mathbf S)_{\kappa\kappa}
            \qquad\text{with}\quad \mathbf P^\text{TS} = \mathbf C^T \cdot
            \text{diag}(g) \cdot \mathbf C

        Returns
        -------
        Tensor
            Orbital-resolved electronic free energy (G = -TS).

        Note
        ----
        Partitioning scheme is set through SCF config
        (`config.fermi.partition`).
        Defaults to an equal partitioning to all atoms (`"equal"`).
        """
        eps = torch.tensor(
            torch.finfo(self._data.occupation.dtype).eps,
            device=self.device,
            dtype=self.dtype,
        )

        occ = torch.clamp(self._data.occupation, min=eps)
        occ1 = torch.clamp(1 - self._data.occupation, min=eps)
        g_spin = torch.log(occ**occ * occ1**occ1) * self.kt
        g = g_spin.sum(-2)

        mode = self.config.fermi.partition

        # partition to atoms equally
        if mode == labels.FERMI_PARTITION_EQUAL:
            real = real_atoms(self._data.numbers)

            count = real.count_nonzero(dim=-1).unsqueeze(-1)
            g_atomic = torch.sum(g, dim=-1, keepdim=True) / count

            return torch.where(
                real, g_atomic.expand(*real.shape), torch.tensor(0.0, **self.dd)
            )

        # partition to atoms via Mulliken population analysis
        if mode == labels.FERMI_PARTITION_ATOMIC:
            # "electronic entropy" density matrix
            density = get_density(
                self._data.evecs,
                g_spin if self.nspin == 2 else g,
            )
            if self.nspin == 2:
                density = density.sum(-3)

            return mulliken.get_atomic_populations(
                self._data.ints.overlap, density, self._data.ihelp
            )

        raise ValueError(f"Unknown partitioning mode '{mode}'.")

    def iterate_charges(self, charges: Tensor) -> Tensor:
        """
        Perform single self-consistent iteration.

        Parameters
        ----------
        charges : Tensor
            Orbital-resolved partial charges vector.

        Returns
        -------
        Tensor
            New orbital-resolved partial charges vector.
        """
        self._data.iter += 1

        q = Charges.from_tensor(
            charges, self._data.charges, batch_mode=self.config.batch_mode
        )

        # SCF cycle (Q -> V -> Q)
        potential = self.charges_to_potential(q)
        new_charges = self.potential_to_charges(potential)

        return new_charges.as_tensor()

    def iterate_potential(self, potential: Tensor) -> Tensor:
        """
        Perform single self-consistent iteration.

        Parameters
        ----------
        potential: Tensor
            Potential vector for each orbital partial charge.

        Returns
        -------
        Tensor
            New potential vector for each orbital partial charge.
        """
        self._data.iter += 1

        pot = Potential.from_tensor(
            potential, self._data.potential, batch_mode=self.config.batch_mode
        )

        # SCF cycle (V -> Q -> V)
        charges = self.potential_to_charges(pot)
        new_potential = self.charges_to_potential(charges)

        return new_potential.as_tensor()

    def iterate_fockian(self, fockian: Tensor) -> Tensor:
        """
        Perform single self-consistent iteration using the Fock matrix.

        Parameters
        ----------
        fockian : Tensor
            Fock matrix.

        Returns
        -------
        Tensor
            New Fock matrix.
        """
        self._data.iter += 1

        # SCF cycle (F -> P -> Q -> V -> F)
        self._data.density = self.hamiltonian_to_density(fockian)
        charges = self.density_to_charges(self._data.density)
        potential = self.charges_to_potential(charges)
        self._data.hamiltonian = self.potential_to_hamiltonian(potential)

        return self._data.hamiltonian

    @timer_decorator("Potential", "SCF")
    def charges_to_potential(self, charges: Charges) -> Potential:
        """
        Compute the potential from the orbital charges.

        Parameters
        ----------
        charges : Tensor
            Orbital-resolved partial charges vector.

        Returns
        -------
        Tensor
            Potential vector for each orbital partial charge.
        """
        potential = self.interactions.get_potential(
            self._data.cache, charges, self._data.ihelp
        )

        self._data.potential = potential.layout

        return potential

    def potential_to_charges(self, potential: Potential) -> Charges:
        """
        Compute the orbital charges from the potential.

        Parameters
        ----------
        potential : Tensor
            Potential vector for each orbital partial charge.

        Returns
        -------
        Tensor
            Orbital-resolved partial charges vector.
        """
        self._data.density = self.potential_to_density(potential)
        return self.density_to_charges(self._data.density)

    def potential_to_density(self, potential: Potential) -> Tensor:
        """
        Obtain the density matrix from the potential.

        Parameters
        ----------
        potential : Tensor
            Potential vector for each orbital partial charge.

        Returns
        -------
        Tensor
            Density matrix.
        """

        self._data.hamiltonian = self.potential_to_hamiltonian(potential)
        return self.hamiltonian_to_density(self._data.hamiltonian)

    @timer_decorator("Charges", "SCF")
    def density_to_charges(self, density: Tensor) -> Charges:
        """
        Compute the orbital charges from the density matrix.

        Parameters
        ----------
        density : Tensor
            Density matrix.

        Returns
        -------
        Tensor
            Orbital-resolved partial charges vector.
        """
        ints = self._data.ints

        density_total = density if self.nspin == 1 else density.sum(-3)
        self._data.energy = einsum(
            "...ik,...ki->...i", density_total, ints.hcore
        )

        charges = Charges(
            mono=mulliken.get_mulliken_orbital_charges(
                ints.overlap,
                density,
                self._data.n0,
                nspin=self.nspin,
            ),
            batch_mode=self.config.batch_mode,
            nspin=self.nspin,
        )

        # Atomic dipole moments (dipole charges)
        if ints.dipole is not None:
            # Again, the diagonal is directly calculated instead of full matrix
            # ("...ik,...mkj->...ijm") as `torch.diagonal` behaves weirdly for
            # more than 2D tensors. Additionally, we move the multipole
            # dimension to the back, which is required for the reduction to
            # atom-resolution.
            dipint = (
                ints.dipole if self.nspin == 1 else ints.dipole.unsqueeze(-4)
            )
            dipole = self._data.ihelp.reduce_orbital_to_atom(
                -einsum("...ik,...mki->...im", density, dipint),
                extra=True,
                dim=-2,
            )
            charges.dipole = (
                dipole
                if self.nspin == 1
                else spin.updown_to_charge_magnetization(dipole, dim=-3)
            )

        # Atomic quadrupole moments (quadrupole charges)
        if ints.quadrupole is not None:
            quadint = (
                ints.quadrupole
                if self.nspin == 1
                else ints.quadrupole.unsqueeze(-4)
            )
            quad = self._data.ihelp.reduce_orbital_to_atom(
                -einsum("...ik,...mki->...im", density, quadint),
                extra=True,
                dim=-2,
            )
            charges.quad = (
                quad
                if self.nspin == 1
                else spin.updown_to_charge_magnetization(quad, dim=-3)
            )

        return charges

    @timer_decorator("Fock build", "SCF")
    def potential_to_hamiltonian(self, potential: Potential) -> Tensor:
        """
        Compute the Hamiltonian from the potential.

        Parameters
        ----------
        potential : Tensor
            Potential vector for each orbital partial charge.

        Returns
        -------
        Tensor
            Hamiltonian matrix.
        """

        def add_vmp_to_h1(h1: Tensor, mpint: Tensor, vmp: Tensor) -> Tensor:
            # spread potential to orbitals
            v = self._data.ihelp.spread_atom_to_orbital(vmp, dim=-2, extra=True)

            # Form dot product over the the multipolar components.
            #  - shape multipole integral: (..., x, norb, norb)
            #  - shape multipole potential: (..., norb, x)
            tmp = 0.5 * einsum("...kij,...jk->...ij", mpint, v)
            return h1 - (tmp + tmp.mT)

        def build(h1: Tensor, channel: Potential) -> Tensor:
            if channel.mono is not None:
                v = channel.mono.unsqueeze(-1) + channel.mono.unsqueeze(-2)
                h1 = h1 - (0.5 * self._data.ints.overlap * v)

            if channel.dipole is not None:
                dpint = self._data.ints.dipole
                if dpint is not None:
                    h1 = add_vmp_to_h1(h1, dpint, channel.dipole)

            if channel.quad is not None:
                qpint = self._data.ints.quadrupole
                if qpint is not None:
                    h1 = add_vmp_to_h1(h1, qpint, channel.quad)
            return h1

        if self.nspin == 1:
            return build(self._data.ints.hcore, potential)

        # H_alpha/beta = H_charge +/- H_magnetization
        hcharge = build(self._data.ints.hcore, potential.select_channel(0))
        hmagnet = build(
            torch.zeros_like(self._data.ints.hcore),
            potential.select_channel(1),
        )
        return torch.stack((hcharge + hmagnet, hcharge - hmagnet), dim=-3)

    def hamiltonian_to_density(self, hamiltonian: Tensor) -> Tensor:
        """
        Compute the density matrix from the Hamiltonian.

        Parameters
        ----------
        hamiltonian : Tensor
            Hamiltonian matrix.

        Returns
        -------
        Tensor
            Density matrix.
        """

        self._data.evals, self._data.evecs = self.diagonalize(hamiltonian)

        # round to integers to avoid numerical errors
        nel = self._data.occupation.sum(-1).round()

        # Restricted orbitals are shared by alpha and beta; unrestricted
        # orbital energies already carry their physical channel axis.
        emo = (
            self._data.evals.unsqueeze(-2).expand([*nel.shape, -1])
            if self.nspin == 1
            else self._data.evals
        )
        mask = self._data.ihelp.spread_shell_to_orbital(
            self._data.ihelp.orbitals_per_shell
        )
        mask = mask.unsqueeze(-2).expand([*nel.shape, -1])

        # Fermi smearing only for non-zero electronic temperature (0.1 K * K2AU)
        if self.kt is not None and not torch.all(self.kt < 3e-7):
            self._data.occupation = filling.get_fermi_occupation(
                nel,
                emo,
                kt=self.kt,
                mask=mask,
                maxiter=self.config.fermi.maxiter,
                thr=self.config.fermi.thresh,
            )

            # check if number of electrons is still correct
            _nel = self._data.occupation.sum(-1)
            if torch.any(torch.abs(nel - _nel.round(decimals=3)) > 1e-4):
                raise RuntimeError(
                    f"Number of electrons changed during Fermi smearing "
                    f"({nel} -> {_nel})."
                )

        occupation = (
            self._data.occupation.sum(-2)
            if self.nspin == 1
            else self._data.occupation
        )
        return get_density(self._data.evecs, occupation)

    @property
    def shape(self) -> torch.Size:
        """
        Returns the shape of the density matrix in this engine.
        """
        return self._data.ints.hcore.shape

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the tensors in this engine.
        """
        return self._data.ints.hcore.dtype

    @property
    def device(self) -> torch.device:
        """
        Returns the device of the tensors in this engine.
        """
        return self._data.ints.hcore.device

    @property
    def dd(self) -> DD:
        """
        Returns the device of the tensors in this engine.
        """
        return {"device": self.device, "dtype": self.dtype}

    def _print(self, charges: Charges, energy: Tensor) -> None:
        # explicitly check to avoid some superfluos calculations
        # if OutputHandler.verbosity < 3:
        # return None

        if charges.mono.ndim < 2:  # pragma: no cover
            energy = self.get_energy(charges).sum(-1)
            _energy = energy.detach().clone()

            ediff = (
                (_energy - self._data.old_energy.sum(-1))
                if self._data.iter > 0
                else 0.0
            )

            density = self._data.density.detach().clone()
            pnorm = (
                torch.linalg.matrix_norm(density - self._data.old_density)
                if self._data.iter > 0
                else 0.0
            )

            _q = charges.mono.detach().clone()
            qdiff = (
                torch.linalg.vector_norm(_q - self._data.old_charges)
                if self._data.iter > 0
                else 0.0
            )

            OutputHandler.write_row(
                "SCF Iterations",
                f"{self._data.iter:3}",
                [
                    f"{_energy: .14E}",
                    f"{ediff: .6E}",
                    f"{pnorm: .6E}",
                    f"{qdiff: .6E}",
                ],
            )

            self._data.old_energy = _energy
            self._data.old_charges = _q
            self._data.old_density = density
        else:
            energy = self.get_energy(charges).detach().clone()
            ediff = (
                torch.linalg.norm(energy - self._data.old_energy)
                if self._data.iter > 0
                else 0.0
            )

            density = self._data.density.detach().clone()
            pnorm = (
                torch.linalg.norm(density - self._data.old_density)
                if self._data.iter > 0
                else 0.0
            )

            _q = charges.mono.detach().clone()
            qdiff = (
                torch.linalg.norm(_q - self._data.old_charges)
                if self._data.iter > 0
                else 0.0
            )

            OutputHandler.write_row(
                "SCF Iterations",
                f"{self._data.iter:3}",
                [
                    f"{energy.norm(): .14E}",
                    f"{ediff: .6E}",
                    f"{pnorm: .6E}",
                    f"{qdiff: .6E}",
                ],
            )

            self._data.old_energy = energy
            self._data.old_charges = _q
            self._data.old_density = density
