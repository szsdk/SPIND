from dataclasses import dataclass, field
from typing import Union

import h5py
import numpy as np


@dataclass
class Solution:
    """
    Indexing solution.
    """

    nb_peaks: int = 0
    centering: str = "P"
    match_rate: float = 0.0
    total_score: float = 0.0
    seed_error: float = 0.0
    centering_score: float = 0.0
    pair_dist: float = float("inf")

    pair_ids: np.ndarray = field(default_factory=lambda: np.empty((0,), int))
    transform_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), float)
    )
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), float))
    # hkl
    hkls: np.ndarray = field(default_factory=lambda: np.empty((0, 3), float))
    rhkls: np.ndarray = field(default_factory=lambda: np.empty((0, 3), float))
    ehkls: np.ndarray = field(default_factory=lambda: np.empty((0, 3), float))

    def write_h5(self, d: Union[h5py.File, h5py.Group]):
        d.attrs["nb_peaks"] = int(self.nb_peaks)
        d.attrs["centering"] = str(self.centering)
        d.attrs["match_rate"] = float(self.match_rate)
        d.attrs["total_score"] = float(self.total_score)
        d.attrs["seed_error"] = float(self.seed_error)
        d.attrs["centering_score"] = float(self.centering_score)
        d.attrs["pair_dist"] = float(self.pair_dist)
        d.create_dataset("pair_ids", data=self.pair_ids)
        d.create_dataset("transform_matrix", data=self.transform_matrix)
        d.create_dataset("rotation_matrix", data=self.rotation_matrix)
        d.create_dataset("hkls", data=self.hkls)
        d.create_dataset("rhkls", data=self.rhkls)
        d.create_dataset("ehkls", data=self.ehkls)

    @staticmethod
    def read_h5(d: Union[h5py.File, h5py.Group]):
        return Solution(
            d.attrs["nb_peaks"],
            d.attrs["centering"],
            d.attrs["match_rate"],
            d.attrs["total_score"],
            d.attrs["seed_error"],
            d.attrs["centering_score"],
            d.attrs["pair_dist"],
            d["pair_ids"][...],
            d["transform_matrix"][...],
            d["rotation_matrix"][...],
            d["hkls"][...],
            d["rhkls"][...],
            d["ehkls"][...],
        )
