from pathlib import Path
import os
import logging

TABLE_HEIGHT = .625

BANDU_ROOT = Path(os.path.dirname(os.path.dirname(__file__)))

TABLE_X_MIN, TABLE_X_MAX = [-1/2 + .2, 1/2 - .2]
TABLE_Y_MIN, TABLE_Y_MAX = [-1.5/2 + .25, 1.5/2 - .25]


bandu_logger = logging.getLogger("bandu")
