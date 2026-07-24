# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
"""Integrity and provenance checks for frozen GFN0 numerical oracles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REFS = Path(__file__).parent / "refs"
EXPECTED_TOTALS = {
    "H2": -1.150088142708,
    "H2O": -4.366814051917,
    "LYS_xao": -45.003398823421,
    "LiH": -0.994315793446,
    "MB16_43_01": -28.497588460796,
    "MB16_43_02": -23.254292745575,
    "SiH4": -4.097876348630,
    "ZnOOH-": -9.338781428029,
}


def test_manifest_and_fixture_contract() -> None:
    manifest = json.loads((REFS / "manifest.json").read_text())
    assert set(manifest["fixtures"]) == {
        f"{system}.json" for system in EXPECTED_TOTALS
    }

    for filename, digest in manifest["fixtures"].items():
        path = REFS / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        fixture = json.loads(path.read_text())
        system = fixture["system"]
        assert fixture["energies"]["total"] == pytest.approx(
            EXPECTED_TOTALS[system]
        )
        assert len(fixture["numbers"]) == len(fixture["positions"])
        assert len(fixture["numbers"]) == len(fixture["atomic_charges"])
        assert fixture["orbitals"]

        metadata = fixture["metadata"]
        assert metadata["oracle"] == "xtb"
        assert metadata["oracle_version"] == "6.7.1"
        assert metadata["oracle_revision"] == "edcfbbe"
        assert metadata["legacy_conversion_constants_used"] is True
        assert metadata["ev_per_hartree"] == 27.21138505
        assert metadata["angstrom_per_bohr"] == 0.52917726
        assert metadata["spin"] == 0


@pytest.mark.parametrize("system", EXPECTED_TOTALS)
def test_component_sum_identity(system: str) -> None:
    fixture = json.loads((REFS / f"{system}.json").read_text())
    energy = fixture["energies"]
    component_sum = sum(
        energy[key]
        for key in ("eht", "fermi", "ies", "d4_2body", "repulsion", "srb")
    )
    # Components and totals are independently printed to twelve decimals.
    assert component_sum == pytest.approx(energy["total"], abs=5.0e-12)
