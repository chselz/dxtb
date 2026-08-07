# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
"""GFN0-xTB Hamiltonian tests against frozen xTB 6.7.1 references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from tad_mctc import storch

from dxtb import IndexHelper, labels
from dxtb._src.integral.factory import new_hcore, new_hcore_gfn0
from dxtb._src.integral.wrappers import hcore, overlap
from dxtb._src.param import GFN0_XTB
from dxtb._src.typing import Tensor
from dxtb._src.xtb.gfn0 import GFN0Hamiltonian

REFS = Path(__file__).parents[1] / "test_gfn0" / "refs"


def load_fixture(name: str) -> dict[str, Any]:
    """Load one frozen GFN0 reference fixture."""
    return json.loads((REFS / f"{name}.json").read_text())


def build_reference(
    name: str,
) -> tuple[dict[str, Any], Tensor, Tensor, GFN0Hamiltonian, Tensor, Tensor]:
    """Build the generalized GFN0 eigenproblem for a frozen fixture."""
    fixture = load_fixture(name)
    numbers = torch.tensor(fixture["numbers"])
    positions = torch.tensor(fixture["positions"], dtype=torch.float64)
    ihelp = IndexHelper.from_numbers(numbers, GFN0_XTB)
    hamiltonian = new_hcore(numbers, GFN0_XTB, ihelp, dtype=torch.float64)
    assert isinstance(hamiltonian, GFN0Hamiltonian)

    ovlp = overlap(
        numbers,
        positions,
        GFN0_XTB,
        driver=labels.INTDRIVER_ANALYTICAL,
    )
    h0 = hamiltonian.build(
        positions, ovlp, charge=fixture["metadata"]["total_charge"]
    )
    return fixture, numbers, positions, hamiltonian, ovlp, h0


def test_factory_and_wrapper_route_gfn0() -> None:
    """Select GFN0 through both integral factory entry points."""
    fixture = load_fixture("H2")
    numbers = torch.tensor(fixture["numbers"])
    positions = torch.tensor(fixture["positions"], dtype=torch.float64)
    ihelp = IndexHelper.from_numbers(numbers, GFN0_XTB)

    assert isinstance(
        new_hcore(numbers, GFN0_XTB, ihelp, dtype=torch.float64),
        GFN0Hamiltonian,
    )
    assert isinstance(
        new_hcore_gfn0(numbers, ihelp, dtype=torch.float64),
        GFN0Hamiltonian,
    )
    matrix = hcore(
        numbers,
        positions,
        GFN0_XTB,
        charge=0,
        driver=labels.INTDRIVER_ANALYTICAL,
    )
    assert matrix.shape == (4, 4)


def test_change_type_preserves_index_tensors() -> None:
    """Convert all GFN0-owned tensors while retaining index tensor dtypes."""
    fixture = load_fixture("H2")
    numbers = torch.tensor(fixture["numbers"])
    ihelp = IndexHelper.from_numbers(numbers, GFN0_XTB)
    hamiltonian = new_hcore_gfn0(
        numbers, ihelp, GFN0_XTB, dtype=torch.float64
    ).type(torch.float32)

    assert hamiltonian.dtype == torch.float32
    assert hamiltonian.selfenergy.dtype == torch.float32
    assert hamiltonian.kq.dtype == torch.float32
    assert hamiltonian.eeq_model.dtype == torch.float32
    assert hamiltonian.numbers.dtype == torch.long
    assert hamiltonian.valence.dtype == torch.bool


def test_charge_is_required() -> None:
    """Require a molecular charge because H0 contains main EEQ shifts."""
    _, _, positions, hamiltonian, ovlp, _ = build_reference("H2")
    with pytest.raises(ValueError, match="charge is required"):
        hamiltonian.build(positions, ovlp)


@pytest.mark.parametrize("name", ["H2O", "LiH", "ZnOOH-"])
def test_main_eeq_charges(name: str) -> None:
    """Reproduce neutral and charged main-model EEQ charges locally."""
    fixture, _, positions, hamiltonian, _, _ = build_reference(name)
    cn = hamiltonian.get_coordination_number(positions)
    charges = hamiltonian.get_eeq_charges(
        positions, fixture["metadata"]["total_charge"], cn
    )
    expected = torch.tensor(fixture["atomic_charges"], dtype=torch.float64)
    assert torch.allclose(charges, expected, atol=5.0e-8, rtol=0.0)
    assert charges.sum() == pytest.approx(
        fixture["metadata"]["total_charge"], abs=1.0e-12
    )


@pytest.mark.parametrize("name", ["H2", "H2O", "SiH4", "ZnOOH-"])
def test_frozen_orbital_energies(name: str) -> None:
    """Match duplicate-H, main-group, and transition-metal eigenvalues."""
    fixture, _, _, _, ovlp, h0 = build_reference(name)
    energies, _ = storch.eighb(h0, ovlp)

    for orbital in fixture["orbitals"]:
        index = orbital["index"] - 1
        assert energies[index] == pytest.approx(orbital["energy"], abs=1.0e-7)


@pytest.mark.parametrize("name", ["H2", "H2O", "LiH", "ZnOOH-"])
def test_occupation_and_eht_sum(name: str) -> None:
    """Use frozen occupations to reproduce the summed EHT band energy."""
    fixture, _, _, _, ovlp, h0 = build_reference(name)
    energies, _ = storch.eighb(h0, ovlp)
    occupation = torch.zeros_like(energies)
    for orbital in fixture["orbitals"]:
        occupation[orbital["index"] - 1] = orbital["occupation"]

    assert occupation.sum().item() == pytest.approx(
        sum(item["occupation"] for item in fixture["orbitals"])
    )
    assert torch.sum(occupation * energies).item() == pytest.approx(
        fixture["energies"]["eht"], abs=5.0e-7
    )


def test_exact_diagonal_and_duplicate_hydrogen_shells() -> None:
    """Set exact self-energies and no same-atom duplicate-shell coupling."""
    fixture, numbers, positions, hamiltonian, _, h0 = build_reference("H2")
    cn = hamiltonian.get_coordination_number(positions)
    charges = hamiltonian.get_eeq_charges(
        positions, fixture["metadata"]["total_charge"], cn
    )
    selfenergy = hamiltonian.get_selfenergy(cn, charges)
    diagonal = hamiltonian.ihelp.spread_shell_to_orbital(selfenergy)

    assert torch.equal(torch.diagonal(h0), diagonal)
    # Each H has 1s/2s; those on-site off-diagonal blocks are exactly zero.
    assert h0[0, 1].item() == 0.0
    assert h0[2, 3].item() == 0.0
    assert numbers.tolist() == [1, 1]


def test_analytical_gradient_rejected() -> None:
    """Fail clearly instead of returning an incomplete GFN0 H0 gradient."""
    _, _, positions, hamiltonian, ovlp, h0 = build_reference("H2")
    with pytest.raises(NotImplementedError, match="GFN0Hamiltonian"):
        hamiltonian.get_gradient(
            positions, ovlp, torch.zeros_like(h0), h0, h0, None, h0.diagonal()
        )
