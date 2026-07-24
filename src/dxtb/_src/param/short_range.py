# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Parametrization: Short-range corrections
========================================

Definitions of short-range corrections that are part of an xTB
parametrization.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict

__all__ = ["ShortRange", "ShortRangeBond"]


class ShortRangeBond(BaseModel):
    """Parameters of the GFN0 short-range bond (SRB) correction."""

    model_config = ConfigDict(extra="forbid")

    shift: float
    """Additional offset for approximate reference bond lengths."""

    prefactor: float
    """Energy prefactor in Hartree."""

    steepness: float
    """Steepness of the Pauling-EN-dependent Gaussian."""

    enscale: float
    """Scaling factor applied to the Pauling electronegativity difference."""

    period1: List[float]
    """First four-entry period polynomial."""

    period2: List[float]
    """Second four-entry period polynomial."""

    cutoff2: float
    """Squared pair cutoff in Bohr squared."""


class ShortRange(BaseModel):
    """Available short-range corrections."""

    model_config = ConfigDict(extra="forbid")

    srb: Optional[ShortRangeBond] = None
    """Short-range bond correction."""
