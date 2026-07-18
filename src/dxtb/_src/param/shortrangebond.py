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
Parametrization: Short-ranged Bond Correction
=============================================

Definition of the GFN0 short-ranged bond (SRB) correction parameters.
"""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["ShortRangeBond"]


class ShortRangeBond(BaseModel):
    """Global parameters for the GFN0 SRB correction."""

    shift: float
    """Additional offset for the reference bond radii."""

    prefactor: float
    """Global scaling factor for the SRB energy contribution."""

    steepness: float
    """Steepness of the Gaussian damping in the SRB term."""

    enscale: float
    """Scaling factor for EN differences in the steepness expression."""

    row1: list[float]
    """EN polynomial coefficients for row-group 1 (Z <= 2)."""

    row2: list[float]
    """EN polynomial coefficients for row-group 2 (2 < Z <= 10)."""

    row3: list[float]
    """EN polynomial coefficients for row-group 3 (10 < Z <= 18)."""

    row4: list[float]
    """EN polynomial coefficients for row-group 4 (Z > 18)."""

    cutoff2: float = 200.0
    """Squared real-space cutoff for SRB interactions."""

    cnmax: float = 8.0
    """Upper clipping value for CN entering SRB radii."""
