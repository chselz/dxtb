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
Short-ranged Bond Correction: Class
==============================

This module implements the short-ranged bond correction class. The
:class:`dxtb.components.Shortranged` class is constructed similar to the
:class:`dxtb.components.Repulsion` class.
"""

from __future__ import annotations

import torch
from tad_mctc.batch import pack
from tad_mctc.convert import any_to_tensor
from tad_mctc.data.radii import ATOMIC_RADII
from tad_mctc.typing import Any, Tensor, override

from dxtb import IndexHelper
from dxtb._src.constants import xtb

from ..base import Classical, ClassicalCache, ComponentCache

__all__ = ["Shortranged", "LABEL_SHORTRANGED"]


LABEL_SHORTRANGED = "Shortranged"
"""
Label for the :class:`.Shortranged` component, coinciding with the class name.
"""


class Shortranged(ClassicalCache):
    """ "Cache for the short-ranged bond parameters."""

    srb: Tensor
    """Short ranged bonds strength"""


class Shortranged(Classical):
    """Representation of the short-ranged bond correction"""
