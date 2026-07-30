# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
"""GFN0-xTB electronegativity-scaled repulsion tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import IndexHelper
from dxtb._src.components.classicals.repulsion import (
    GFN0Repulsion,
    new_repulsion,
)
from dxtb._src.param import GFN0_XTB

REFS = Path(__file__).parents[2] / "test_gfn0" / "refs"


def load_reference(
    system: str,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    data = json.loads((REFS / f"{system}.json").read_text())
    numbers = torch.tensor(data["numbers"])
    positions = torch.tensor(data["positions"], dtype=torch.float64)
    return data, numbers, positions


@pytest.mark.parametrize(
    "system",
    ["H2", "H2O", "LiH", "MB16_43_01", "ZnOOH-", "LYS_xao"],
)
def test_reference(system: str) -> None:
    """Reproduce the frozen GFN0 repulsion components."""
    data, numbers, positions = load_reference(system)
    repulsion = new_repulsion(
        torch.unique(numbers), GFN0_XTB, dtype=positions.dtype
    )
    assert isinstance(repulsion, GFN0Repulsion)
    assert repulsion.label == "Repulsion"

    ihelp = IndexHelper.from_numbers(numbers, GFN0_XTB)
    atomwise = repulsion.get_energy(
        positions, repulsion.get_cache(numbers, ihelp)
    )

    assert atomwise.shape == numbers.shape
    assert atomwise.sum() == pytest.approx(
        data["energies"]["repulsion"], abs=1.0e-11
    )


def test_pair_formula_and_cutoff() -> None:
    """Check EN²/EN⁴ scaling and the parameterized 40 Bohr cutoff."""
    numbers = torch.tensor([5, 6])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64
    )
    repulsion = new_repulsion(
        torch.unique(numbers), GFN0_XTB, dtype=positions.dtype
    )
    assert isinstance(repulsion, GFN0Repulsion)
    cache = repulsion.get_cache(
        numbers, IndexHelper.from_numbers(numbers, GFN0_XTB)
    )

    element_b = GFN0_XTB.element["B"]
    element_c = GFN0_XTB.element["C"]
    den2 = (element_b.en - element_c.en) ** 2
    alpha = (element_b.arep * element_c.arep) ** 0.5
    alpha *= 1.0 + (0.01 * den2 + 0.01 * den2**2) * -0.09
    expected = (
        element_b.zeff * element_c.zeff * math.exp(-alpha * 2.0**1.5) / 2.0
    )

    assert repulsion.get_energy(positions, cache).sum() == pytest.approx(
        expected, abs=1.0e-14
    )

    distant = positions.clone()
    distant[1, 0] = 40.01
    assert repulsion.get_energy(distant, cache).sum() == 0.0


def test_batch_padding_and_autograd() -> None:
    """Preserve packed padding and the natural position-gradient graph."""
    references = [load_reference(name) for name in ("H2O", "ZnOOH-")]
    numbers = pack([reference[1] for reference in references])
    positions = pack([reference[2] for reference in references])
    assert isinstance(numbers, torch.Tensor)
    assert isinstance(positions, torch.Tensor)
    positions.requires_grad_(True)

    repulsion = new_repulsion(
        torch.unique(numbers), GFN0_XTB, dtype=positions.dtype
    )
    assert isinstance(repulsion, GFN0Repulsion)
    cache = repulsion.get_cache(
        numbers, IndexHelper.from_numbers(numbers, GFN0_XTB)
    )
    atomwise = repulsion.get_energy(positions, cache)

    assert atomwise.sum(-1).detach() == pytest.approx(
        [reference[0]["energies"]["repulsion"] for reference in references],
        abs=1.0e-11,
    )
    assert torch.count_nonzero(atomwise[0, 3:]) == 0

    (gradient,) = torch.autograd.grad(atomwise.sum(), positions)
    assert torch.isfinite(gradient).all()


def test_no_custom_analytical_gradient() -> None:
    """Do not select the unscaled legacy analytical-gradient class."""
    with pytest.raises(NotImplementedError, match="GFN0 repulsion"):
        new_repulsion(
            torch.tensor([1]),
            GFN0_XTB,
            with_analytical_gradient=True,
        )
