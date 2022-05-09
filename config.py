import os
from pathlib import Path
import logging

BANDU_ROOT = Path(os.path.dirname(__file__))

TABLE_X_MIN, TABLE_X_MAX = [-1/2 + .2, 1/2 - .2]
TABLE_Y_MIN, TABLE_Y_MAX = [-1.5/2 + .25, 1.5/2 - .25]

# TABLE_X_MIN, TABLE_X_MAX = [-1/2 + .25, 1/2 + .15]
# TABLE_Y_MIN, TABLE_Y_MAX = [-.25, .25]

TABLE_HEIGHT = .625

bandu_logger = logging.getLogger("bandu")


def get_object_height(name):
    if name == "Bandu Block":
        object_height = 0.10029600089788437
    elif name == "Skewed Rectangular Prism":
        object_height = 0.18886199867725373
    else:
        object_height = 0
        print("ln22 object height = 0 need to fix")
        # raise NotImplementedError
    return object_height