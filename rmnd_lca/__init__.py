
__all__ = (
    'InventorySet',
    'DatabaseCleaner',
    'RemindDataCollection',
    'NewDatabase',
    'Electricity',

)
__version__ = (0, 0, 1)

# For relative imports to work in Python 3.6
import os, sys;
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

from .activity_maps import InventorySet
from .clean_datasets import DatabaseCleaner
from .data_collection import RemindDataCollection
from .ecoinvent_modification import NewDatabase
from .electricity import Electricity
