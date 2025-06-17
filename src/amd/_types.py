"""Contains the FloatArray and IntArray types."""

from typing import Union

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[Union[np.float32, np.float64]]
UIntArray = npt.NDArray[Union[np.uint32, np.uint64]]
AtomTypeArray = npt.NDArray[Union[np.uint8, np.uint16, np.uint32, np.uint64]]