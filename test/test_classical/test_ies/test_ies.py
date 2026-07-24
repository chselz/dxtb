# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
"""Tests for the GFN0-xTB isotropic electrostatic contribution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB, GFN2_XTB, IndexHelper
from dxtb._src.components.classicals import (
    IES,
    LABEL_IES,
    ClassicalList,
    IESCache,
    new_ies,
)
from dxtb._src.components.classicals.base import ClassicalCache
from dxtb._src.param import GFN0_XTB, ParamModule

REFS = Path(__file__).parents[2] / "test_gfn0" / "refs"

# Independent legacy GFN0 CN values. These exercise the old Å/Bohr conversion,
# the 40 Bohr cutoff, erf counting with k=7.5, and the smooth cap at 8.
REFERENCE_CN = {
    "H2O": [
        1.987322860680810,
        0.994147318210111,
        0.994147318210111,
    ],
    "ZnOOH-": [
        1.981426052327369,
        0.996158951014089,
        1.974526793898510,
        0.999659384748928,
    ],
    "MB16_43_01": [
        4.018215605239292,
        0.972247058984081,
        1.984876324993608,
        1.471998526294176,
        0.996978315972177,
        0.996288827142561,
        1.450787467004970,
        1.990550537844816,
        3.830423105115959,
        1.001851281573857,
        0.996142033754135,
        1.923091087219516,
        4.587000105110676,
        3.804891127880177,
        3.940050009026652,
        5.271440631634270,
    ],
}


def load_reference(
    system: str, dtype: torch.dtype = torch.float64
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Load a self-contained Phase 0 reference fixture."""
    data = json.loads((REFS / f"{system}.json").read_text())
    numbers = torch.tensor(data["numbers"])
    positions = torch.tensor(data["positions"], dtype=dtype)
    return data, numbers, positions


@pytest.mark.parametrize("system", ["H2O", "ZnOOH-", "MB16_43_01"])
def test_reference(system: str) -> None:
    """Reproduce capped CN, main EEQ charges, and atomwise/summed IES."""
    data, numbers, positions = load_reference(system)
    charge = torch.tensor(
        float(data["metadata"]["total_charge"]), dtype=positions.dtype
    )

    ies = new_ies(numbers, GFN0_XTB, dtype=positions.dtype)
    assert isinstance(ies, IES)
    assert ies.label == LABEL_IES

    ihelp = IndexHelper.from_numbers(numbers, GFN0_XTB)
    cache = ies.get_cache(numbers, ihelp)
    assert isinstance(cache, IESCache)

    cn = ies.get_coordination_number(positions, cache)
    charges, atomwise = cache.model.solve(
        numbers,
        positions,
        charge,
        cn,
        return_energy=True,
    )
    energy = ies.get_energy(positions, cache, charge=charge)

    assert cn == pytest.approx(REFERENCE_CN[system], abs=2.0e-14)
    assert charges == pytest.approx(
        data["atomic_charges"],
        abs=data["metadata"]["tolerances"]["charge_atol"],
    )
    assert charges.sum() == pytest.approx(charge, abs=5.0e-15)
    assert energy.shape == numbers.shape
    assert energy == pytest.approx(atomwise, abs=1.0e-15)
    assert energy.sum() == pytest.approx(
        data["energies"]["ies"],
        abs=data["metadata"]["tolerances"]["component_atol"],
    )


def test_batch_and_padding() -> None:
    """Packed charged batches conserve charge and preserve zero padding."""
    references = [load_reference(name) for name in ("H2O", "ZnOOH-")]
    data = [reference[0] for reference in references]
    numbers = pack([reference[1] for reference in references])
    positions = pack([reference[2] for reference in references])
    assert isinstance(numbers, torch.Tensor)
    assert isinstance(positions, torch.Tensor)
    charge = torch.tensor([[0.0], [-1.0]], dtype=positions.dtype)

    ies = new_ies(numbers, GFN0_XTB, dtype=positions.dtype)
    assert isinstance(ies, IES)
    cache = ies.get_cache(numbers)
    cn = ies.get_coordination_number(positions, cache)
    charges, _ = cache.model.solve(
        numbers,
        positions,
        charge,
        cn,
        return_energy=True,
    )
    energy = ies.get_energy(positions, cache, charge=charge)

    assert energy.sum(-1) == pytest.approx(
        [reference["energies"]["ies"] for reference in data],
        abs=1.0e-8,
    )
    assert charges.sum(-1, keepdim=True) == pytest.approx(charge, abs=5.0e-15)
    assert torch.count_nonzero(energy[0, 3:]) == 0
    assert torch.count_nonzero(charges[0, 3:]) == 0
    assert torch.count_nonzero(cn[0, 3:]) == 0


@pytest.mark.parametrize(
    ("dtype", "atol"),
    [(torch.float32, 2.0e-6), (torch.float64, 1.0e-8)],
)
def test_dtype(dtype: torch.dtype, atol: float) -> None:
    """Preserve the requested floating-point type and observed accuracy."""
    data, numbers, positions = load_reference("H2O", dtype)
    ies = new_ies(numbers, GFN0_XTB, dtype=dtype)
    assert isinstance(ies, IES)
    assert ies.dtype == dtype

    cache = ies.get_cache(numbers)
    energy = ies.get_energy(positions, cache, charge=0)
    assert energy.dtype == dtype
    assert energy.sum() == pytest.approx(data["energies"]["ies"], abs=atol)


def test_position_autograd() -> None:
    """Keep the natural Torch graph through CN and the EEQ solve."""
    _, numbers, positions = load_reference("H2O")
    positions.requires_grad_(True)
    ies = new_ies(numbers, GFN0_XTB, dtype=positions.dtype)
    assert isinstance(ies, IES)

    energy = ies.get_energy(positions, ies.get_cache(numbers), charge=0).sum()
    (gradient,) = torch.autograd.grad(energy, positions)

    assert torch.isfinite(gradient).all()
    assert torch.linalg.vector_norm(gradient) > 0
    assert gradient.sum(0) == pytest.approx(torch.zeros(3), abs=1.0e-14)


def test_classical_list_label() -> None:
    """Expose IES under its stable key through the standard list contract."""
    data, numbers, positions = load_reference("H2O")
    ies = new_ies(numbers, GFN0_XTB, dtype=positions.dtype)
    assert isinstance(ies, IES)

    classicals = ClassicalList(ies, dtype=positions.dtype)
    caches = classicals.get_cache(
        numbers, IndexHelper.from_numbers(numbers, GFN0_XTB)
    )
    energies = classicals.get_energy(
        positions, caches, charge=torch.tensor([0.0])
    )

    assert set(energies) == {LABEL_IES}
    assert energies[LABEL_IES].sum() == pytest.approx(
        data["energies"]["ies"], abs=1.0e-8
    )


def test_factory_selection_and_param_module() -> None:
    """Only an EEQ block selects IES, for both parameter representations."""
    numbers = torch.tensor([8, 1, 1])
    assert new_ies(numbers, GFN1_XTB) is None
    assert new_ies(numbers, GFN2_XTB) is None
    assert isinstance(new_ies(numbers, ParamModule(GFN0_XTB)), IES)


def test_factory_rejects_unknown_cn() -> None:
    """Fail clearly instead of silently using a different CN convention."""
    par = GFN0_XTB.model_copy(deep=True)
    assert par.charge is not None
    assert par.charge.eeq is not None
    par.charge.eeq.cn = "exp"

    with pytest.raises(ValueError, match="only supports erf"):
        new_ies(torch.tensor([1]), par)


def test_cache_and_errors() -> None:
    """Exercise cache reuse, invalidation, and invalid-call diagnostics."""
    _, numbers, positions = load_reference("H2O")
    ies = new_ies(numbers, GFN0_XTB, dtype=positions.dtype)
    assert isinstance(ies, IES)

    cache = ies.get_cache(numbers)
    assert ies.get_cache(numbers) is cache
    assert ies.cache_is_latest((numbers.detach().clone(),))

    with pytest.raises(ValueError, match="charge is required"):
        ies.get_energy(positions, cache)
    with pytest.raises(TypeError, match="IESCache"):
        ies.get_energy(
            positions,
            ClassicalCache(dtype=positions.dtype),
            charge=0,
        )

    ies.reset()
    assert ies.cache is None
