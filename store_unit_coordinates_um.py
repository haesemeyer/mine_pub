
"""
Script that processes a folder with experiment hdf5 files and for each experiment creates a csv file
that contains <x,y,z> um coordinates of all caiman unit centroids
"""

from typing import Any
import argparse
import os
from os import path
from experiment import Experiment2P
import numpy as np


class CheckArgs(argparse.Action):
    """
    Check our command line arguments for validity
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values: Any, option_string=None):
        if self.dest == 'folder':
            if not path.exists(values):
                raise argparse.ArgumentError(self, "Specified directory does not exist")
            if not path.isdir(values):
                raise argparse.ArgumentError(self, "The destination is a file but should be a directory")
            setattr(namespace, self.dest, values)
        else:
            raise Exception("Parser was asked to check unknown argument")


def process_experiment(data_dir: str, exp_file: str) -> None:
    out_name = exp_file[:exp_file.find(".hdf5")] + ".csv"
    out_file_name = path.join(data_dir, out_name)
    coordinates_um = []
    with Experiment2P(path.join(data_dir, exp_file)) as exp:
        # the z-step is fixed for the whole experiment
        res_z = exp.info_data["z_step"]
        for p in range(exp.n_planes):
            # compute this plane's x and y resolution from the fov (um) and the scan size
            res_x = exp.scanner_data[p]["fov"] / exp.scanner_data[p]["pixels_per_line"]
            res_y = exp.scanner_data[p]["fov"] / exp.scanner_data[p]["lines_per_image"]
            # create vector with proper broadcast dimension to convert plane centroid coordinates into um
            um_transform = (np.r_[res_x, res_y, res_z])[None, :]
            coordinates = exp.all_centroids[p]
            plane = np.full(coordinates.shape[0], p)[:, None]
            # add z-coordinate
            coordinates = np.c_[coordinates, plane]
            coordinates_um.append(coordinates * um_transform)
    coordinates_um = np.vstack(coordinates_um)
    np.savetxt(out_file_name, coordinates_um, delimiter=' ', fmt="%.1f")


def main(data_dir: str) -> None:
    """
    Runs the main loop of the script
    :param data_dir: The directory with the closed-loop data
    :return: None
    """
    file_list = [f for f in os.listdir(data_dir) if ".hdf5" in f]
    for f in file_list:
        process_experiment(data_dir, f)


if __name__ == "__main__":

    a_parser = argparse.ArgumentParser(prog="store_unit_coordinates_um",
                                       description="Script that processes a folder with experiment hdf5 files and for "
                                                   "each experiment creates a csv file that contains <x,y,z> um "
                                                   "coordinates of all caiman unit centroids")
    a_parser.add_argument("-f", "--folder", help="Path to folder with experiment hdf5 files", type=str, default="",
                          action=CheckArgs)

    args = a_parser.parse_args()

    data_folder = args.folder

    main(data_folder)
