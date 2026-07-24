# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
"""Load the built-in GFN0-xTB parametrization lazily."""

from __future__ import annotations

from pathlib import Path

from dxtb._src.loader.lazy import LazyLoaderParam as Lazy

from ..base import Param

__all__ = ["GFN0_XTB"]


GFN0_XTB: Param = Lazy(Path(__file__).parent / "gfn0-xtb.toml")  # type: ignore
