"""Array aliases shared across the package.

Ref: Sec. 6.5 (the descriptor computations are dense volumetric array work).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type IntArray = npt.NDArray[np.int64]
type Bool3D = npt.NDArray[np.bool_]
type Float3D = npt.NDArray[np.float64]
