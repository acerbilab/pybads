import json
from collections import namedtuple
from pathlib import Path

import numpy as np

from pybads.utils.iteration_history import IterationHistory


class BADSDump:
    """
    This class is responsible for storing a dump of BADS in json format.
    It includes the information of the optimization solution and the iteration history of the algorithm.
    """

    def __init__(self, file_name, dir_path=None):
        if dir_path is None:
            dir_path = "./dumps/"

        Path(dir_path).mkdir(exist_ok=True)
        self.file_path = dir_path + file_name

    def set_attributes(
        self,
        x: np.ndarray,
        u: np.ndarray,
        fval,
        fsd,
        iteration_history: IterationHistory,
        x_true_global_min,
    ):
        self.x = x.tolist()
        self.u = u.tolist()
        self.fval = fval
        self.fsd = fsd
        self.iteration_history = {
            "u": list(map(lambda u: u.tolist(), iteration_history["u"])),
            "fval": iteration_history["fval"].tolist(),
            "fsd": iteration_history["fsd"].tolist(),
            "mesh_size": iteration_history["mesh_size"].tolist(),
            "search_mesh_size": iteration_history["search_mesh_size"].tolist(),
            "yval": iteration_history["yval"].tolist(),
            "x_true_global_min": x_true_global_min.tolist(),
            "init_N": iteration_history["init_N"].tolist(),
            "ntrain": iteration_history["ntrain"].tolist(),
        }

    def to_JSON(
        self,
        x: np.ndarray,
        u: np.ndarray,
        fval,
        fsd,
        iteration_history,
        x_true_global_min,
    ):
        self.set_attributes(
            x, u, fval, fsd, iteration_history, x_true_global_min
        )

        json_object = json.dumps(self, default=lambda o: o.__dict__, indent=4)
        with open(self.file_path, "w") as outfile:
            outfile.write(json_object)
            outfile.close()

    def load_JSON(
        self,
    ):
        jsonObject = None
        with open(self.file_path) as jsonFile:
            jsonObject = json.load(jsonFile)
            jsonFile.close()
        return jsonObject


def objectDecoder(object):
    return namedtuple("X", object.keys())(*object.values())
