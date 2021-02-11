from typing import Callable

import xarray

Transform = Callable[[xarray.DataArray], xarray.DataArray]
