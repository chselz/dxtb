# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
"""GFN0-xTB isotropic electrostatic energy."""

from __future__ import annotations

import torch
from tad_mctc.convert import any_to_tensor
from tad_mctc.ncoord import coordination_number, erf_count
from tad_multicharge.model.eeq import EEQModel

from dxtb import IndexHelper
from dxtb._src.typing import Any, Tensor, override

from ..base import Classical, ClassicalCache, ComponentCache

__all__ = ["IES", "IESCache", "LABEL_IES"]


LABEL_IES = "IES"
"""Stable label for the GFN0 isotropic electrostatic contribution."""


class IESCache(ClassicalCache):
    """Coordinate-independent data for the GFN0 EEQ solve."""

    numbers: Tensor
    """Atomic numbers, including batch padding."""

    model: EEQModel
    """GFN0 main electronegativity-equilibration model."""

    rcov: Tensor
    """Atom-resolved D3 covalent radii from tad-mctc."""

    __slots__ = ["numbers", "model", "rcov"]

    def __init__(
        self,
        numbers: Tensor,
        model: EEQModel,
        rcov: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            device=device if device is not None else rcov.device,
            dtype=dtype if dtype is not None else rcov.dtype,
        )
        self.numbers = numbers
        self.model = model
        self.rcov = rcov


class IES(Classical):
    """
    GFN0-xTB isotropic electrostatic energy from the main EEQ model.

    The EEQ charges are transient inputs to this classical energy evaluation;
    they are neither cached nor exposed as calculator result charges.
    """

    chi: Tensor
    """Element-resolved electronegativity lookup table."""

    eeq_kcn: Tensor
    """Element-resolved coordination-number dependence."""

    eta: Tensor
    """Element-resolved chemical hardness."""

    rad: Tensor
    """Element-resolved Gaussian charge width."""

    rcov: Tensor
    """D3 covalent-radius lookup table from tad-mctc."""

    cutoff: Tensor
    """Coordination-number real-space cutoff."""

    cn_max: Tensor
    """Smooth coordination-number cap."""

    cn_kcn: Tensor
    """Steepness of the erf counting function."""

    __slots__ = [
        "chi",
        "eeq_kcn",
        "eta",
        "rad",
        "rcov",
        "cutoff",
        "cn_max",
        "cn_kcn",
    ]

    def __init__(
        self,
        chi: Tensor,
        eeq_kcn: Tensor,
        eta: Tensor,
        rad: Tensor,
        rcov: Tensor,
        cutoff: Tensor | float | int,
        cn_max: Tensor | float | int,
        cn_kcn: Tensor | float | int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.chi = chi.to(**self.dd)
        self.eeq_kcn = eeq_kcn.to(**self.dd)
        self.eta = eta.to(**self.dd)
        self.rad = rad.to(**self.dd)
        self.rcov = rcov.to(**self.dd)
        self.cutoff = any_to_tensor(cutoff, **self.dd)
        self.cn_max = any_to_tensor(cn_max, **self.dd)
        self.cn_kcn = any_to_tensor(cn_kcn, **self.dd)

    @override
    def get_cache(
        self, numbers: Tensor, ihelp: IndexHelper | None = None, **_: Any
    ) -> IESCache:
        """Construct or reuse coordinate-independent EEQ data."""
        cachvars = (numbers.detach().clone(),)

        if self.cache_is_latest(cachvars):
            if not isinstance(self.cache, IESCache):
                raise TypeError(
                    f"Cache in {self.label} is not of type 'IESCache'."
                )
            return self.cache

        if numbers.numel() > 0 and int(numbers.max().item()) >= len(self.chi):
            raise ValueError(
                "Atomic numbers exceed the element range used to construct IES."
            )

        self._cachevars = cachvars
        model = EEQModel(
            self.chi,
            self.eeq_kcn,
            self.eta,
            self.rad,
            **self.dd,
        )
        self.cache = IESCache(numbers, model, self.rcov[numbers], **self.dd)
        return self.cache

    def get_coordination_number(
        self, positions: Tensor, cache: IESCache
    ) -> Tensor:
        """Evaluate capped GFN0 coordination numbers."""
        return coordination_number(
            cache.numbers,
            positions,
            counting_function=erf_count,
            rcov=cache.rcov,
            cutoff=self.cutoff,
            cn_max=self.cn_max,
            kcn=self.cn_kcn,
        )

    @override
    def get_energy(
        self,
        positions: Tensor,
        cache: ComponentCache,
        charge: Tensor | float | int | None = None,
        **_: Any,
    ) -> Tensor:
        """Calculate the atomwise GFN0 isotropic electrostatic energy."""
        if not isinstance(cache, IESCache):
            raise TypeError(f"Cache in {self.label} is not of type 'IESCache'.")
        if charge is None:
            raise ValueError("Total molecular charge is required for IES.")

        total_charge = any_to_tensor(
            charge,
            device=positions.device,
            dtype=positions.dtype,
        )
        cn = self.get_coordination_number(positions, cache)
        _charges, energy = cache.model.solve(
            cache.numbers,
            positions,
            total_charge,
            cn,
            return_energy=True,
        )
        return energy
