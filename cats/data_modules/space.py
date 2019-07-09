import numpy as np
from .dataset import dataset
from .data_interface import data_tellurics

class space(data_tellurics):
    def get_tellurics(self, **data):
        return 1