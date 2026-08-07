# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
"""GFN0 basis/index construction tests."""

from __future__ import annotations

import pytest
import torch

from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.basis.ortho import gaussian_integral
from dxtb._src.param import GFN0_XTB


def test_all_supported_elements_build_basis() -> None:
    numbers = torch.arange(1, 87, dtype=torch.long)
    ihelp = IndexHelper.from_numbers(numbers, GFN0_XTB)
    basis = Basis(numbers, GFN0_XTB, ihelp, dtype=torch.double)
    alphas, coeffs = basis.create_cgtos()

    assert len(alphas) == len(coeffs) == ihelp.unique_angular.numel()
    assert len(alphas) == sum(
        len(element.shells) for element in GFN0_XTB.element.values()
    )
    assert all(torch.isfinite(value).all() for value in (*alphas, *coeffs))


def test_gfn0_hydrogen_shells_are_orthogonalized() -> None:
    numbers = torch.tensor([1], dtype=torch.long)
    ihelp = IndexHelper.from_numbers(numbers, GFN0_XTB)
    basis = Basis(numbers, GFN0_XTB, ihelp, dtype=torch.double)
    alphas, coeffs = basis.create_cgtos()

    overlap = gaussian_integral(alphas[0], alphas[1], coeffs[0], coeffs[1])
    # The contracted coefficients are normalized numerically; the residual is
    # below 1e-9 for the float64 STO-nG expansion.
    assert overlap.item() == pytest.approx(0.0, abs=1.0e-9)


def test_padding_and_element_limit() -> None:
    padded = torch.tensor([[1, 0], [86, 1]], dtype=torch.long)
    ihelp = IndexHelper.from_numbers(padded, GFN0_XTB)
    assert ihelp.shells_per_atom[0, 1].item() == 0
