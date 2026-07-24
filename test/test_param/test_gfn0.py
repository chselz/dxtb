# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
"""Tests for the built-in GFN0-xTB parameterization."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from dxtb._src.components.interactions.coulomb import new_es2
from dxtb._src.param import GFN0_XTB, Param, ParamModule

PARAMETER_FILE = (
    Path(__file__).parents[2] / "src/dxtb/_src/param/gfn0/gfn0-xtb.toml"
)
PARAMETER_SHA256 = (
    "a8aa7847808e6bc96a2a09c572d30a35951d203fde65de6d6fda6dbdcf28c226"
)


def test_load_complete_parameterization() -> None:
    assert GFN0_XTB.meta is not None
    assert GFN0_XTB.meta.name == "GFN0-xTB"
    assert len(GFN0_XTB.element) == 86
    assert GFN0_XTB.charge is not None
    assert GFN0_XTB.charge.eeq is not None
    assert GFN0_XTB.charge.effective is None
    assert GFN0_XTB.short_range is not None
    assert GFN0_XTB.short_range.srb is not None

    assert GFN0_XTB.charge.eeq.cutoff == 40.0
    assert GFN0_XTB.charge.eeq.cn_max == 8.0
    assert GFN0_XTB.charge.eeq.kcn == 7.5
    assert GFN0_XTB.hamiltonian is not None
    assert GFN0_XTB.hamiltonian.xtb.enshell == [0.6, -0.1, -0.2, 0.0]
    assert GFN0_XTB.hamiltonian.xtb.kdiff == pytest.approx(1.1241)
    assert GFN0_XTB.hamiltonian.xtb.enscale4 == 4.0


@pytest.mark.parametrize(
    ("symbol", "shells", "level", "kqat", "eeq_chi"),
    [
        ("H", ["1s", "2s"], -11.9223639, 0.2473983, 1.25),
        ("He", ["1s", "2p"], -20.9532631, -0.7787934, 1.2912463),
        ("Sc", ["3d", "4s", "4p"], -8.7129730, -0.6961569, 0.9641451),
        ("La", ["5d", "6s", "6p"], -8.1422087, -0.7252284, 0.9335398),
        ("Au", ["5d", "6s", "6p"], -12.1009764, 0.5364732, 1.3555382),
        ("Rn", ["6s", "6p", "5d"], -13.6425470, -0.0946010, 1.2653792),
    ],
)
def test_element_sentinels(
    symbol: str,
    shells: list[str],
    level: float,
    kqat: float,
    eeq_chi: float,
) -> None:
    element = GFN0_XTB.element[symbol]
    assert element.shells == shells
    assert element.levels[0] == pytest.approx(level)
    assert element.kqat == pytest.approx(kqat)
    assert element.eeq_chi == pytest.approx(eeq_chi)
    assert len(element.kq or []) == len(shells)
    assert len(element.ngauss) == len(shells)
    assert all(value > 0 for value in element.ngauss)


def test_pair_and_srb_sentinels() -> None:
    assert GFN0_XTB.hamiltonian is not None
    pair = GFN0_XTB.hamiltonian.xtb.kpair
    assert len(pair) == 861
    assert pair["Sc-Sc"] == pytest.approx(1.1)
    assert pair["Cu-Cu"] == pytest.approx(0.9)
    assert pair["Ag-Cu"] == pytest.approx(0.9)
    assert pair["Au-Ag"] == pytest.approx(0.9)

    assert GFN0_XTB.short_range is not None
    assert GFN0_XTB.short_range.srb is not None
    srb = GFN0_XTB.short_range.srb
    assert srb.period1 == pytest.approx(
        [29.84522887, -1.70549806, 6.54013762, 6.39169003]
    )
    assert srb.period2 == pytest.approx(
        [-8.87843763, 2.10878369, 0.08009374, -0.85808076]
    )
    assert srb.cutoff2 == 200.0


def test_artifact_fingerprint() -> None:
    assert (
        hashlib.sha256(PARAMETER_FILE.read_bytes()).hexdigest()
        == PARAMETER_SHA256
    )


def test_param_module_roundtrip() -> None:
    module = ParamModule(GFN0_XTB, dtype=torch.double)
    assert module.to_pydantic() == GFN0_XTB._loaded  # type: ignore[attr-defined]


@pytest.mark.parametrize("suffix", [".toml", ".json", ".yaml"])
def test_serialization_roundtrip(suffix: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"gfn0{suffix}"
        try:
            GFN0_XTB.to_file(path)
        except ImportError:
            pytest.skip(f"Optional writer for {suffix} is unavailable.")
        restored = Param.from_file(path)
    assert restored == GFN0_XTB._loaded  # type: ignore[attr-defined]


def test_method_specific_validation() -> None:
    data = GFN0_XTB.clean_model_dump()
    del data["element"]["Rn"]
    with pytest.raises(ValidationError, match="H--Rn"):
        Param(**data)

    data = GFN0_XTB.clean_model_dump()
    data["element"]["H"]["kq"] = [0.1]
    with pytest.raises(ValidationError, match="one value per shell"):
        Param(**data)

    data = GFN0_XTB.clean_model_dump()
    data["element"]["H"]["eeq_chi"] = float("nan")
    with pytest.raises(ValidationError, match="not finite"):
        Param(**data)

    data = GFN0_XTB.clean_model_dump()
    data["element"]["H"]["unknown_gfn0_value"] = 1.0
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Param(**data)


def test_eeq_does_not_activate_es2() -> None:
    unique = torch.tensor([1, 8], dtype=torch.long)
    assert new_es2(unique, GFN0_XTB, dtype=torch.double) is None
