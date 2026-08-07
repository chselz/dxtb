# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
"""GFN0-xTB two-body-only D4 tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tad_dftd4.dispersion import TwoBodyTerm
from tad_mctc.batch import pack

from dxtb._src.components.classicals.dispersion import (
    DispersionD4GFN0,
    new_dispersion,
)
from dxtb._src.components.classicals.gfn0 import (
    GFN0_D4_GA,
    GFN0_D4_GC,
    GFN0_D4_WF,
    gfn0_d4_coordination_number,
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
    ["H2", "H2O", "LiH", "SiH4", "MB16_43_01", "ZnOOH-", "LYS_xao"],
)
def test_reference(system: str) -> None:
    """Reproduce frozen neutral and charged two-body D4 components."""
    data, numbers, positions = load_reference(system)
    dispersion = new_dispersion(numbers, GFN0_XTB, dtype=positions.dtype)
    assert isinstance(dispersion, DispersionD4GFN0)
    assert dispersion.label == "DispersionD4"
    cache = dispersion.get_cache(numbers)

    atomwise = dispersion.get_energy(
        positions,
        cache,
        charge=data["metadata"]["total_charge"],
    )
    assert atomwise.shape == numbers.shape
    assert atomwise.sum() == pytest.approx(
        data["energies"]["d4_2body"], abs=1.0e-8
    )


def test_intermediate_sentinels() -> None:
    """Pin internally generated H2O D4 EEQ charges and covalent CN."""
    _, numbers, positions = load_reference("H2O")
    dispersion = new_dispersion(numbers, GFN0_XTB, dtype=positions.dtype)
    assert isinstance(dispersion, DispersionD4GFN0)
    cache = dispersion.get_cache(numbers)
    assert cache.eeq_model is not None

    main_cn = dispersion.get_coordination_number(positions, cache)
    charges = cache.eeq_model.solve(
        numbers, positions, torch.tensor(0.0, dtype=positions.dtype), main_cn
    )
    d4_cn = gfn0_d4_coordination_number(numbers, positions)

    assert charges == pytest.approx(
        [-0.5940268763262897, 0.2970134381631447, 0.2970134381631449],
        abs=2.0e-14,
    )
    assert charges.sum() == pytest.approx(0.0, abs=5.0e-15)
    assert d4_cn == pytest.approx(
        [1.6104536495298425, 0.8052268247649212, 0.8052268247649212],
        abs=2.0e-14,
    )


def test_explicit_model_and_two_body_only() -> None:
    """Pin D4 defaults and ensure no ATM term is registered."""
    _, numbers, _ = load_reference("H2O")
    dispersion = new_dispersion(numbers, GFN0_XTB, dtype=torch.float64)
    assert isinstance(dispersion, DispersionD4GFN0)
    cache = dispersion.get_cache(numbers)

    assert cache.model.ga == GFN0_D4_GA == 3.0
    assert cache.model.gc == GFN0_D4_GC == 2.0
    assert cache.model.wf == pytest.approx(GFN0_D4_WF)
    assert cache.model.ref_charges == "eeq"
    assert cache.cutoff.disp2 == pytest.approx(60.0)
    assert cache.cutoff.cn == pytest.approx(40.0)
    assert cache.cutoff.cn_eeq == pytest.approx(40.0)

    evaluator = dispersion.get_dispersion(cache)
    assert len(evaluator.terms) == 1
    assert isinstance(evaluator.terms[0], TwoBodyTerm)


def test_runtime_charge_and_charge_dependence() -> None:
    """Construct without charge, require it at evaluation, and use its value."""
    _, numbers, positions = load_reference("ZnOOH-")
    dispersion = new_dispersion(numbers, GFN0_XTB, dtype=positions.dtype)
    assert isinstance(dispersion, DispersionD4GFN0)
    cache = dispersion.get_cache(numbers)

    with pytest.raises(ValueError, match="charge is required"):
        dispersion.get_energy(positions, cache)
    with pytest.raises(ValueError, match="internally"):
        dispersion.get_energy(
            positions,
            cache,
            q=torch.zeros_like(numbers, dtype=positions.dtype),
            charge=0,
        )

    neutral = dispersion.get_energy(positions, cache, charge=0)
    anion = dispersion.get_energy(positions, cache, charge=-1)
    assert not torch.allclose(neutral, anion, atol=1.0e-12, rtol=0.0)


def test_batch_padding() -> None:
    """Match separate neutral/charged calculations in a padded batch."""
    references = [load_reference(name) for name in ("H2O", "ZnOOH-")]
    numbers = pack([reference[1] for reference in references])
    positions = pack([reference[2] for reference in references])
    assert isinstance(numbers, torch.Tensor)
    assert isinstance(positions, torch.Tensor)

    dispersion = new_dispersion(numbers, GFN0_XTB, dtype=positions.dtype)
    assert isinstance(dispersion, DispersionD4GFN0)
    atomwise = dispersion.get_energy(
        positions,
        dispersion.get_cache(numbers),
        charge=torch.tensor([[0.0], [-1.0]], dtype=positions.dtype),
    )

    assert atomwise.sum(-1) == pytest.approx(
        [reference[0]["energies"]["d4_2body"] for reference in references],
        abs=1.0e-8,
    )
    assert torch.count_nonzero(atomwise[0, 3:]) == 0
