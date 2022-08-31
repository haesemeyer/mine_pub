"""
Script to build main file of data from reticulospinal backfills - note that this file has much less information
than what is stored for the main rwave experiments

Intended file information:

I RS Fit Neurons - only for neurons which were fit by the ann
<Neuron ID> <Barcode> <is RSpinal> <labelBrightness>

II RS Neuron list - where neurons come from
<Neuron ID> <Exp-ID> <plane> <is ann fit> <is RSpinal>

III Stimuli
<Exp-ID> <plane> <temperatures>

IV Behaviors
<Exp-ID> <plane> <behaviors (separated by predictor)>

V Experiment list
<Exp-ID> <Original Filename>
"""

import rwave_build_main as rbm
import h5py
import numpy as np
import pandas as pd
import argparse
from typing import Any, Tuple
from os import path
import os
from experiment import Experiment2P
import tifffile
import utilities


def get_rs_fit_neurons(datafile: h5py.File) -> pd.DataFrame:
    """
    Gets information about all ann fit neurons in form of a pandas dataframe
    :param datafile: The hdf5 file from which to read the data
    :return: pandas dataframe with the fit neuron data
    """
    keys = ["neuron_id", "barcode", "is_rspinal", "label_brightness"]
    return rbm.dataframe_from_hdf5(datafile, "rs_fit_neurons", keys)


def save_rs_fit_neurons(datafile: h5py.File, neuron_ids: np.ndarray, barcodes: np.ndarray, is_rspinal: np.ndarray,
                        label_brightness: np.ndarray):
    if not rbm.check_arg_sizes(neuron_ids, barcodes, is_rspinal, label_brightness):
        raise ValueError("All inputs must have the same size along dimension 0")
    grp = datafile.create_group("rs_fit_neurons")
    grp.create_dataset("neuron_id", data=neuron_ids)
    grp.create_dataset("barcode", data=barcodes)
    grp.create_dataset("is_rspinal", data=is_rspinal)
    grp.create_dataset("label_brightness", data=label_brightness)


def get_rs_neuron_list(datafile: h5py.File) -> pd.DataFrame:
    """
    Gets basic information (experiment, plane, location, fit status) of all neurons in a pandas dataframe
    :param datafile: The hdf5 file from which to read the data
    :return: dataframe with the rs neuron list
    """
    keys = ["neuron_id", "experiment_id", "plane", "is_ann_fit", "is_rspinal"]
    return rbm.dataframe_from_hdf5(datafile, "rs_neuron_list", keys)


def save_rs_neuron_list(datafile: h5py.File, experiment_ids: np.ndarray, planes: np.ndarray, abt_ann: np.ndarray,
                        is_rspinal: np.ndarray):
    if not rbm.check_arg_sizes(experiment_ids, planes, abt_ann, is_rspinal):
        raise ValueError("All inputs must have the same size along dimension 0")
    grp = datafile.create_group("rs_neuron_list")
    grp.create_dataset("neuron_id", data=np.arange(experiment_ids.size))
    grp.create_dataset("experiment_id", data=experiment_ids)
    grp.create_dataset("plane", data=planes)
    grp.create_dataset("is_ann_fit", data=abt_ann)
    grp.create_dataset("is_rspinal", data=is_rspinal)


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
                raise argparse.ArgumentError(self, f"Specified directory {values} does not exist")
            if not path.isdir(values):
                raise argparse.ArgumentError(self, "The destination is a file but should be a directory")
            setattr(namespace, self.dest, values)
        else:
            raise Exception("Parser was asked to check unknown argument")


def get_data(base_folder: str, e_folder: str, eid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                                 np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get information to analyze for one experiment
    :param base_folder: The base folder with all experiment hdf5 files
    :param e_folder: The analysis folder of the experiment in question
    :param eid: This experiment's unique ID
    :return:
        [0]: The test correlations of all n units in the experiment
        [1]: n x terms (x n_boot) array of taylor component scores
        [2]: n long vector of curvature metrics
        [3]: n long vector of experiment IDs
        [4]: n long vector of nlc metric scores
        [5]: n long vector of backfill channel brightness
        [6]: n long vector of plane ids
        [7]: n long vector of average segmentation value (fraction of neuron pixels part of segm. RS neuron)
    """
    datafile_path = path.join(base_folder, path.join(e_folder, f"{e_folder}_ANN_analysis.hdf5"))
    bf_channel = []
    plane_ids = []
    with h5py.File(datafile_path, 'r') as dfile:
        test_corrs = dfile["correlations_test"][()]
        taylor_scores = dfile["taylor_by_pred"][()]
        curve_metrics = dfile["avg_curvatures"][()]
        nlc_metrics = dfile["nlc_metrics"][()]
        n_planes = dfile["n_planes"][()]
        for p_id in range(n_planes):
            plane_group = dfile[f"{p_id}"]
            bf_plane = plane_group["anatomy_brightness"][()]
            plane_ids.append(np.full(bf_plane.size, p_id))
            bf_channel.append(bf_plane)
    bf_channel = np.hstack(bf_channel)
    plane_ids = np.hstack(plane_ids)
    exp_ids = np.full(curve_metrics.size, eid)

    orig_file_path = path.join(base_folder, e_folder+".hdf5")  # the CAIMAN fit file
    with Experiment2P(orig_file_path, False) as exp:
        seg_values = []
        for p_id in range(exp.n_planes):
            # load the corresponding mask file
            label_file_path = path.join(path.join(path.join(base_folder, e_folder), "labeled"),
                                        e_folder+f"_Z_{p_id}_1labeled.tif")
            p = tifffile.imread(label_file_path).astype(float)
            # loop over all components, calculating the average label value
            # <component-ix, weight, x-coord, y-coord>
            n_components = int(np.max(exp.all_spatial[p_id][:, 0]) + 1)  # component indices are 0-based
            br = np.zeros(n_components, dtype=np.float32)
            for j in range(n_components):
                this_component = exp.all_spatial[p_id][:, 0].astype(int) == j
                spatial_x = exp.all_spatial[p_id][this_component, 2].astype(int)
                spatial_y = exp.all_spatial[p_id][this_component, 3].astype(int)
                br[j] = np.mean(p[spatial_y, spatial_x])
            seg_values.append(br)
    seg_values = np.hstack(seg_values)

    return test_corrs, taylor_scores, curve_metrics, exp_ids, nlc_metrics, bf_channel, plane_ids, seg_values


def main(folder: str, c_thresh: float, sig_thresh: float, nonlin_thresh: float, outfile: str):
    """
    Runs analysis
    :param folder: The folder with experiment hdf5 files and ann analysis subfolders
    :param c_thresh: The correlation threshold above which to consider units
    :param sig_thresh: The taylor metric threshold - metrics not significantly above this will be set to 0
    :param nonlin_thresh: The threshold for considering a neuron nonlinear
    :param outfile: The name of the output file
    """
    experiment_ids, experiment_names = [], []  # to relate experiment ids with their respective names
    # get list of ANN analysis subfolders
    folder_list = os.listdir(folder)
    sub_folder_list = [f for f in folder_list if path.isdir(path.join(folder, f)) and "RSpinal" in f]
    # Create dictionary (and then dataframe) to save per-experiment information
    # for our overview sheet
    info_dict = {"Experiment name": [], "N Caiman": [], "N Fit": []}
    # the ann test trial correlations
    all_test_corrs = []
    # the raw taylor scores which include bootstrap information
    all_taylor_scores_raw = []
    # the curvature metrics
    all_curve_metrics = []
    # the experiment ids for each neuron
    all_exp_ids = []
    # the plane that each neuron came from
    all_plane_ids = []
    # the nlc values
    all_nlc_metrics = []
    # the backfill channel brightness of each neuron
    all_bf_brightness = []
    # the average segmentation of each neuron
    all_rs_seg = []
    for i, flder in enumerate(sub_folder_list):
        info_dict["Experiment name"].append(flder)
        experiment_ids.append(i)
        experiment_names.append(flder.encode())
        tc, ts, cm, ei, nlc, bf, pids, segm = get_data(folder, flder, i)
        info_dict["N Caiman"].append(np.sum(np.isfinite(tc)))
        info_dict["N Fit"].append(np.sum(tc >= c_thresh))
        all_test_corrs.append(tc)
        all_taylor_scores_raw.append(ts)
        all_curve_metrics.append(cm)
        all_exp_ids.append(ei)
        all_plane_ids.append(pids)
        all_nlc_metrics.append(nlc)
        all_bf_brightness.append(bf)
        all_rs_seg.append(segm)

    info_dframe = pd.DataFrame(info_dict)
    info_dframe.to_csv(path.join(folder, f"Overview_Info_{path.splitext(outfile)[0]}.csv"))

    # convert lists into appropriate arrays
    all_test_corrs = np.hstack(all_test_corrs)
    all_bf_brightness = np.hstack(all_bf_brightness)
    all_taylor_scores_raw = np.vstack(all_taylor_scores_raw)
    # calculate the required corrected significance to get an overall .05 score then threshold
    # taylor metrics based on that
    min_significance = 1 - 0.05 / np.sum(all_test_corrs >= c_thresh)
    normal_quantiles_by_sigma = np.array([0.682689492137, 0.954499736104, 0.997300203937, 0.999936657516,
                                          0.999999426697, 0.999999998027])
    n_sigma = np.where((min_significance - normal_quantiles_by_sigma) < 0)[0][0] + 1
    taylor_sig: np.ndarray = all_taylor_scores_raw[:, :, 0] - n_sigma * all_taylor_scores_raw[:, :, 1] - sig_thresh
    all_taylor_scores: np.ndarray = all_taylor_scores_raw[:, :, 0]
    all_taylor_scores[taylor_sig <= 0] = 0
    all_curve_metrics = np.hstack(all_curve_metrics)
    all_nlc_metrics = np.hstack(all_nlc_metrics)
    all_exp_ids = np.hstack(all_exp_ids)
    all_plane_ids = np.hstack(all_plane_ids)
    all_rs_seg = np.hstack(all_rs_seg)

    above_thresh_ann = all_test_corrs >= corr_th
    is_rspinal = all_rs_seg >= 0.5  # at least half the neuron is covered by a mask
    ann_fit_neuron_ids = np.arange(all_test_corrs.size)[above_thresh_ann]

    nonlin_lrm = utilities.NonlinClassifier.get_standard_model()
    nl_probs = nonlin_lrm.nonlin_probability(all_curve_metrics[above_thresh_ann], all_nlc_metrics[above_thresh_ann])
    nonl_probabilities = np.full_like(all_curve_metrics, np.nan)
    nonl_probabilities[above_thresh_ann] = nl_probs

    is_nonlinear = nl_probs > nonlin_thresh

    # extract barcode labels
    # load taylor metric names from file and prepend "Nonlin"
    labels = [b"Nonlin"]
    datafile_path = path.join(folder, path.join(sub_folder_list[0], f"{sub_folder_list[0]}_ANN_analysis.hdf5"))
    with h5py.File(datafile_path, 'r') as dfile:
        taylor_names = dfile["taylor_names"][()]
        labels += [tn[0] for tn in taylor_names]

    # perform barcoding
    ats_abt = all_taylor_scores[above_thresh_ann]
    barcode = np.c_[is_nonlinear[:, None], ats_abt > 0]

    # extract names of behavior predictors from labels
    with h5py.File(datafile_path, 'r') as dfile:
        predictor_names = dfile["predictor_names"][()]
    behavior_names = predictor_names[1:]

    with h5py.File(os.path.join(folder, outfile), "w") as main_file:
        main_file.create_dataset("ANN test correlation threshold", data=c_thresh)
        main_file.create_dataset("Taylor threshold", data=sig_thresh)
        main_file.create_dataset("Nonlinearity threshold", data=nonlin_thresh)
        main_file.create_dataset("Taylor corrected significance", data=(1-min_significance))
        main_file.create_dataset("Taylor n_sigma to reach significance", data=n_sigma)
        main_file.create_dataset("Barcode labels", data=np.vstack(labels))
        main_file.create_dataset("Behavior names", data=behavior_names)
        save_rs_neuron_list(main_file, all_exp_ids, all_plane_ids, above_thresh_ann, is_rspinal)
        rbm.save_experiment_list(main_file, np.hstack(experiment_ids), np.hstack(experiment_names))
        save_rs_fit_neurons(main_file, ann_fit_neuron_ids, barcode, is_rspinal[above_thresh_ann],
                            all_bf_brightness[above_thresh_ann])
        s, p, e = rbm.extract_stimuli(folder, sub_folder_list)
        rbm.save_stimuli(main_file, e, p, s)
        b, p, e = rbm.extract_behaviors(folder, sub_folder_list, behavior_names)
        rbm.save_behaviors(main_file, e, p, b)


if __name__ == "__main__":
    a_parser = argparse.ArgumentParser(prog="rspinal_build_main",
                                       description="Will create main file of analysis results for reticulospinal "
                                                   "backfill control data")
    a_parser.add_argument("-f", "--folder", help="Path to folder with experiment hdf5 files", type=str, default="",
                          action=CheckArgs)
    a_parser.add_argument("-co", "--corrthresh", help="The fit test correlation threshold", type=float,
                          default=np.sqrt(0.5))
    a_parser.add_argument("-si", "--sigthresh", help="The threshold for taylor metric significance", type=float,
                          default=0.1)
    a_parser.add_argument("-nl", "--nlthresh", help="The threshold of nonlinearity", type=float, default=0.5)
    a_parser.add_argument("-o", "--output", help="The name of the output file", default="rs_analysis.hdf5")

    cl_args = a_parser.parse_args()

    data_folder = cl_args.folder
    corr_th = cl_args.corrthresh
    sig_th = cl_args.sigthresh
    nl_th = cl_args.nlthresh
    ofile = cl_args.output

    main(data_folder, corr_th, sig_th, nl_th, ofile)
