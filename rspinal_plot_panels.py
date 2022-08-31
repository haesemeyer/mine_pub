"""
Script to plot paper figure panels from reticulospinal backfill analysis file generated via rspinal_build_main.py
"""

import argparse
import os
from os import path
from typing import Any
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl
from pandas import DataFrame

import rspinal_build_main as rbm
import upsetplot as ups


class CheckArgs(argparse.Action):
    """
    Check our command line arguments for validity
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values: Any, option_string=None):
        if self.dest == 'file':
            if not path.exists(values):
                raise argparse.ArgumentError(self, f"Specified file {values} does not exist")
            if not path.isfile(values):
                raise argparse.ArgumentError(self, f"The destination {values} is not a file")
            setattr(namespace, self.dest, values)
        else:
            raise Exception("Parser was asked to check unknown argument")


if __name__ == "__main__":
    mpl.rcParams['pdf.fonttype'] = 42

    a_parser = argparse.ArgumentParser(prog="rwspinal_plot_panels",
                                       description="Will plot figure panels related to spinal backfill controls")
    a_parser.add_argument("-f", "--file", help="Path to the analysis hdf5 file", type=str, default="",
                          action=CheckArgs)
    a_parser.add_argument("-png", "--save_png", help="If set figures will be saved as png instead of pdf",
                          action='store_true')

    cl_args = a_parser.parse_args()

    data_file = cl_args.file
    as_png = cl_args.save_png

    ext = "png" if as_png else "pdf"

    data_folder, data_filename = path.split(data_file)
    base_name = path.splitext(data_filename)[0]

    plot_dir = path.join(data_folder, f"{base_name}_rspinal_panels")

    if not path.exists(plot_dir):
        os.makedirs(plot_dir)

    with h5py.File(data_file, 'r') as dfile:

        barcode_labels = dfile["Barcode labels"]
        barcode_labels = [bl[0].decode('utf-8') for bl in barcode_labels]

        #######################################
        # Analysis of ANN fit neurons, relating
        # barcode to reticulospinal mask
        #######################################

        df_fit = rbm.get_rs_fit_neurons(dfile)
        barcodes = np.vstack(df_fit["barcode"])
        is_rspinal = np.hstack(df_fit["is_rspinal"])

        # plot barcode clustering upset plot. We reduce the barcodes since we only want to show the
        # relationship between something that is behavior vs. sensory related and the mask label
        nl_id = barcode_labels.index("Nonlin")
        t_id = barcode_labels.index("Temperature")
        behav_id = barcode_labels.index("behavior")
        ang_id = barcode_labels.index("sum_tail")
        disp_id = barcode_labels.index("rolling_vigor")
        start_id = barcode_labels.index("bout_start")

        barcodes_to_use = np.zeros((np.sum(is_rspinal), 2), dtype=bool)
        brsp = barcodes[is_rspinal]
        btu_labels = ["Temperature", "Behavior"]
        barcodes_to_use[:, 0] = brsp[:, t_id]
        barcodes_to_use[:, 1] = np.logical_or(brsp[:, behav_id], brsp[:, ang_id])
        barcodes_to_use[:, 1] = np.logical_or(brsp[:, 2], brsp[:, disp_id])
        barcodes_to_use[:, 1] = np.logical_or(brsp[:, 2], brsp[:, start_id])

        df_test_metric = DataFrame(barcodes_to_use, columns=btu_labels)
        aggregate = ups.from_indicators(df_test_metric)
        fig = pl.figure()
        up_set = ups.UpSet(aggregate, subset_size='count', facecolor="grey", sort_by='cardinality',
                           sort_categories_by=None)
        up_set.style_subsets(present=["Behavior", "Temperature"], facecolor='C0', label="Mixed selectivity")
        up_set.style_subsets(present="Temperature", absent="Behavior", facecolor="C1", label="Temperature")
        up_set.style_subsets(present="Behavior", absent="Temperature", facecolor="w", edgecolor="C2",
                             label="Behavior")
        axes_dict = up_set.plot(fig)
        fig.savefig(path.join(plot_dir, f"Barcode_Rspinal_Plot.{ext}"), dpi=300)
