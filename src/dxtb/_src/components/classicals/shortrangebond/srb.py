# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
"""GFN0-xTB short-range bond correction."""

from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs
from tad_mctc.convert import any_to_tensor
from tad_mctc.data import en as element_en
from tad_mctc.ncoord import coordination_number, erf_count

from dxtb import IndexHelper
from dxtb._src.typing import Any, Tensor, override

from ..base import Classical, ClassicalCache, ComponentCache

__all__ = ["LABEL_SRB", "ShortRangeBond", "ShortRangeBondCache"]


LABEL_SRB = "ShortRangeBond"
"""Stable label for the GFN0 SRB contribution."""


class ShortRangeBondCache(ClassicalCache):
    """Atom-resolved, coordinate-independent SRB data."""

    __slots__ = ["numbers", "r0", "cnfak", "en", "pauling", "rcov"]

    def __init__(
        self,
        numbers: Tensor,
        r0: Tensor,
        cnfak: Tensor,
        en: Tensor,
        pauling: Tensor,
        rcov: Tensor,
    ) -> None:
        super().__init__(device=r0.device, dtype=r0.dtype)
        self.numbers = numbers
        self.r0 = r0
        self.cnfak = cnfak
        self.en = en
        self.pauling = pauling
        self.rcov = rcov


class ShortRangeBond(Classical):
    """GFN0 short-range correction for heteronuclear B--F pairs."""

    __slots__ = [
        "r0",
        "cnfak",
        "en",
        "rcov",
        "shift",
        "prefactor",
        "steepness",
        "enscale",
        "period1",
        "period2",
        "cutoff2",
        "cn_cutoff",
        "cn_max",
        "cn_kcn",
    ]

    def __init__(
        self,
        r0: Tensor,
        cnfak: Tensor,
        en: Tensor,
        rcov: Tensor,
        shift: Tensor | float | int,
        prefactor: Tensor | float | int,
        steepness: Tensor | float | int,
        enscale: Tensor | float | int,
        period1: Tensor,
        period2: Tensor,
        cutoff2: Tensor | float | int,
        cn_cutoff: Tensor | float | int,
        cn_max: Tensor | float | int,
        cn_kcn: Tensor | float | int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.r0 = r0.to(**self.dd)
        self.cnfak = cnfak.to(**self.dd)
        self.en = en.to(**self.dd)
        self.rcov = rcov.to(**self.dd)
        self.shift = any_to_tensor(shift, **self.dd)
        self.prefactor = any_to_tensor(prefactor, **self.dd)
        self.steepness = any_to_tensor(steepness, **self.dd)
        self.enscale = any_to_tensor(enscale, **self.dd)
        self.period1 = period1.to(**self.dd)
        self.period2 = period2.to(**self.dd)
        self.cutoff2 = any_to_tensor(cutoff2, **self.dd)
        self.cn_cutoff = any_to_tensor(cn_cutoff, **self.dd)
        self.cn_max = any_to_tensor(cn_max, **self.dd)
        self.cn_kcn = any_to_tensor(cn_kcn, **self.dd)

    @override
    def get_cache(
        self, numbers: Tensor, ihelp: IndexHelper | None = None, **_: Any
    ) -> ShortRangeBondCache:
        """Construct or reuse atom-resolved SRB parameters."""
        if ihelp is None:
            raise ValueError("IndexHelper must be passed for SRB.")

        cachvars = (numbers.detach().clone(),)
        if self.cache_is_latest(cachvars):
            if not isinstance(self.cache, ShortRangeBondCache):
                raise TypeError(
                    f"Cache in {self.label} is not of type "
                    "'ShortRangeBondCache'."
                )
            return self.cache

        self._cachevars = cachvars
        self.cache = ShortRangeBondCache(
            numbers=numbers,
            r0=ihelp.spread_uspecies_to_atom(self.r0),
            cnfak=ihelp.spread_uspecies_to_atom(self.cnfak),
            en=ihelp.spread_uspecies_to_atom(self.en),
            pauling=element_en.PAULING(**self.dd)[numbers],
            rcov=self.rcov[numbers],
        )
        return self.cache

    @override
    def get_energy(
        self, positions: Tensor, cache: ComponentCache, **_: Any
    ) -> Tensor:
        """Return the symmetrically half-partitioned atomwise SRB energy."""
        if not isinstance(cache, ShortRangeBondCache):
            raise TypeError(
                f"Cache in {self.label} is not of type 'ShortRangeBondCache'."
            )

        cn = coordination_number(
            cache.numbers,
            positions,
            counting_function=erf_count,
            rcov=cache.rcov,
            cutoff=self.cn_cutoff,
            cn_max=self.cn_max,
            kcn=self.cn_kcn,
        )

        eligible = (cache.numbers >= 5) & (cache.numbers <= 9)
        pair_mask = (
            real_pairs(cache.numbers, mask_diagonal=True)
            & eligible.unsqueeze(-1)
            & eligible.unsqueeze(-2)
            & (cache.numbers.unsqueeze(-1) != cache.numbers.unsqueeze(-2))
        )

        eps = torch.tensor(torch.finfo(positions.dtype).eps, **self.dd)
        distances = torch.where(
            pair_mask,
            storch.cdist(positions, positions, p=2),
            eps,
        )
        pair_mask = pair_mask & (distances**2 < self.cutoff2)

        radius = cache.r0 + cache.cnfak * cn + self.shift
        row = torch.where(
            cache.numbers <= 2,
            0,
            torch.where(
                cache.numbers <= 10,
                1,
                torch.where(cache.numbers <= 18, 2, 3),
            ),
        )
        p1 = self.period1[row]
        p2 = self.period2[row]

        fitted_difference = torch.abs(
            cache.en.unsqueeze(-1) - cache.en.unsqueeze(-2)
        )
        k1 = 0.005 * (p1.unsqueeze(-1) + p1.unsqueeze(-2))
        k2 = 0.005 * (p2.unsqueeze(-1) + p2.unsqueeze(-2))
        factor = 1.0 - k1 * fitted_difference - k2 * fitted_difference**2
        reference_distance = (
            radius.unsqueeze(-1) + radius.unsqueeze(-2)
        ) * factor

        pauling_difference = cache.pauling.unsqueeze(
            -1
        ) - cache.pauling.unsqueeze(-2)
        width = self.steepness * (1.0 + self.enscale * pauling_difference**2)
        pair_energy = self.prefactor * torch.exp(
            -width * (distances - reference_distance) ** 2
        )
        pair_energy = torch.where(
            pair_mask,
            pair_energy,
            torch.tensor(0.0, **self.dd),
        )
        return 0.5 * pair_energy.sum(-1)
