# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
"""GFN0-xTB electronegativity-scaled repulsion."""

from __future__ import annotations

import torch
from tad_mctc.batch import real_pairs

from dxtb import IndexHelper
from dxtb._src.typing import Any, Tensor, override

from .base import BaseRepulsionCache
from .rep import Repulsion

__all__ = ["GFN0Repulsion"]


class GFN0Repulsion(Repulsion):
    """GFN0 repulsion with pair-exponent electronegativity scaling."""

    en: Tensor
    """Element-specific electronegativities for unique species."""

    enscale: Tensor
    """Global electronegativity scaling factor."""

    __slots__ = ["en", "enscale"]

    def __init__(
        self,
        arep: Tensor,
        zeff: Tensor,
        en: Tensor,
        kexp: Tensor,
        enscale: Tensor,
        cutoff: Tensor | float | int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            arep,
            zeff,
            kexp,
            klight=None,
            cutoff=cutoff,
            device=device,
            dtype=dtype,
        )
        self.label = "Repulsion"
        self.en = en.to(**self.dd)
        self.enscale = enscale.to(**self.dd)

    @override
    def get_cache(
        self, numbers: Tensor, ihelp: IndexHelper | None = None, **_: Any
    ) -> BaseRepulsionCache:
        """Build the EN-scaled pair exponent and standard repulsion cache."""
        if ihelp is None:
            raise ValueError("IndexHelper must be passed for GFN0 repulsion.")

        cachvars = (numbers.detach().clone(),)
        if self.cache_is_latest(cachvars):
            if not isinstance(self.cache, BaseRepulsionCache):
                raise TypeError(
                    f"Cache in {self.label} is not of type "
                    "'BaseRepulsionCache'."
                )
            return self.cache

        self._cachevars = cachvars

        arep = ihelp.spread_uspecies_to_atom(self.arep)
        zeff = ihelp.spread_uspecies_to_atom(self.zeff)
        en = ihelp.spread_uspecies_to_atom(self.en)
        kexp = ihelp.spread_uspecies_to_atom(
            self.kexp.expand(torch.unique(numbers).shape)
        )

        mask = real_pairs(numbers, mask_diagonal=True)
        eps = torch.finfo(arep.dtype).tiny
        alpha = torch.where(
            mask,
            torch.sqrt(arep.unsqueeze(-1) * arep.unsqueeze(-2) + eps),
            torch.tensor(0.0, **self.dd),
        )

        den2 = (en.unsqueeze(-1) - en.unsqueeze(-2)) ** 2
        alpha = alpha * (1.0 + (0.01 * den2 + 0.01 * den2**2) * self.enscale)

        zpair = zeff.unsqueeze(-1) * zeff.unsqueeze(-2) * mask
        kpair = (
            kexp.unsqueeze(-1) * kexp.new_ones(kexp.shape).unsqueeze(-2) * mask
        )

        self.cache = BaseRepulsionCache(mask, alpha, zpair, kpair)
        return self.cache
