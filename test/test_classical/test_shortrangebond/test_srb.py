# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
"""GFN0-xTB short-range bond correction tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB, GFN2_XTB, IndexHelper
from dxtb._src.components.classicals.shortrangebond import (
    LABEL_SRB,
    ShortRangeBond,
    ShortRangeBondCache,
    new_srb,
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
    ["H2O", "MB16_43_01", "MB16_43_02", "ZnOOH-", "LYS_xao"],
)
def test_reference(system: str) -> None:
    """Reproduce zero and nonzero frozen SRB components."""
    data, numbers, positions = load_reference(system)
    srb = new_srb(torch.unique(numbers), GFN0_XTB, dtype=positions.dtype)
    assert isinstance(srb, ShortRangeBond)
    assert srb.label == LABEL_SRB

    cache = srb.get_cache(numbers, IndexHelper.from_numbers(numbers, GFN0_XTB))
    assert isinstance(cache, ShortRangeBondCache)
    atomwise = srb.get_energy(positions, cache)

    assert atomwise.shape == numbers.shape
    assert atomwise.sum() == pytest.approx(data["energies"]["srb"], abs=5.0e-9)


@pytest.mark.parametrize(
    ("numbers", "distance"),
    [
        ([1, 8], 2.0),
        ([6, 6], 2.0),
        ([5, 6], 200.0**0.5 + 0.01),
    ],
)
def test_pair_selection(numbers: list[int], distance: float) -> None:
    """Reject ineligible, homonuclear, and out-of-cutoff pairs."""
    number_tensor = torch.tensor(numbers)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=torch.float64
    )
    srb = new_srb(torch.unique(number_tensor), GFN0_XTB, dtype=positions.dtype)
    assert isinstance(srb, ShortRangeBond)
    cache = srb.get_cache(
        number_tensor,
        IndexHelper.from_numbers(number_tensor, GFN0_XTB),
    )
    assert srb.get_energy(positions, cache).sum() == 0.0


def test_eligible_pair_and_atomwise_partition() -> None:
    """Evaluate an eligible B--C pair and split it symmetrically."""
    numbers = torch.tensor([5, 6])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64
    )
    srb = new_srb(torch.unique(numbers), GFN0_XTB, dtype=positions.dtype)
    assert isinstance(srb, ShortRangeBond)
    cache = srb.get_cache(numbers, IndexHelper.from_numbers(numbers, GFN0_XTB))
    atomwise = srb.get_energy(positions, cache)

    assert atomwise.sum() < 0
    assert atomwise[0] == pytest.approx(atomwise[1], abs=1.0e-15)


def test_batch_padding_and_autograd() -> None:
    """Handle heterogeneous packed batches without detaching positions."""
    references = [load_reference(name) for name in ("MB16_43_01", "ZnOOH-")]
    numbers = pack([reference[1] for reference in references])
    positions = pack([reference[2] for reference in references])
    assert isinstance(numbers, torch.Tensor)
    assert isinstance(positions, torch.Tensor)
    positions.requires_grad_(True)

    srb = new_srb(torch.unique(numbers), GFN0_XTB, dtype=positions.dtype)
    assert isinstance(srb, ShortRangeBond)
    cache = srb.get_cache(numbers, IndexHelper.from_numbers(numbers, GFN0_XTB))
    atomwise = srb.get_energy(positions, cache)

    assert atomwise.sum(-1).detach() == pytest.approx(
        [reference[0]["energies"]["srb"] for reference in references],
        abs=5.0e-9,
    )
    assert torch.count_nonzero(atomwise[1, 4:]) == 0

    (gradient,) = torch.autograd.grad(atomwise.sum(), positions)
    assert torch.isfinite(gradient).all()


def test_factory_selection() -> None:
    """Leave GFN1/GFN2 unchanged when no SRB block is present."""
    unique = torch.tensor([1, 6])
    assert new_srb(unique, GFN1_XTB) is None
    assert new_srb(unique, GFN2_XTB) is None
