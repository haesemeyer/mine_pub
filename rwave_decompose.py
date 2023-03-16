
"""
Script for performing decomposition to compute predictor contributions and nonlinearity metrics
"""

import os
from os import path

import utilities
from taylorDecomp import taylor_decompose, d2ca_dr2, complexity_scores
import h5py
import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
from typing import Optional, Dict, List, Any
import model
from utilities import create_overwrite, Data
import argparse


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
        elif self.dest == 'processes':
            if values <= 0:
                raise argparse.ArgumentError(self, "The number of processes to use has to be larger than 0.")
            setattr(namespace, self.dest, values)
        elif self.dest == 'corr_thresh':
            if values < -1 or values > 1:
                raise argparse.ArgumentError(self, "corr_thresh has to be between -1 and 1 (inclusive)")
        else:
            raise Exception("Parser was asked to check unknown argument")


def analyze_experiment(base_folder: str, e_folder: str, c_thresh: float, ovr: bool) -> int:
    """
    Analyzes one experiment identified by the folder that contains the fitting hdf5 file as well
    as all model weight files
    :param base_folder: The root of e_folder
    :param e_folder: The folder containing all experiment files
    :param c_thresh: The test correlations threshold to use to define if a unit is processed or not
    :param ovr: If set to true file will not be skipped if info exists and contents will be replaced
    :returns:
        The number of processed (above threshold) units or -1 in case of error
    """
    hist_steps = 5 * 10  # 10s history at 5Hz

    datafile_path = path.join(base_folder, path.join(e_folder, f"{e_folder}_ANN_analysis.hdf5"))
    modelfile_path = path.join(base_folder, path.join(e_folder, f"{e_folder}_fit_models.hdf5"))
    if not path.exists(datafile_path):
        print(f"Error {datafile_path} does not exist")
        return -2
    with h5py.File(datafile_path, 'a') as dfile, h5py.File(modelfile_path, 'r') as model_file:
        # if this file has already been analyzed and we do not intend to overwrite, skip
        if "linear_model_score" in dfile and not ovr:
            return -1
        test_corrs = dfile["correlations_test"][()]
        plane_ids = dfile["plane_ids"][()]
        cell_indices = dfile["cell_indices"][()]
        curr_plane = -1
        init_in: Optional[np.ndarray] = None
        data: Optional[Data] = None
        n_predictors = None
        lin_model_scores = np.full(test_corrs.size, np.nan)  # for each neuron the score of the linear approximation
        taylor_full = np.full_like(lin_model_scores, np.nan)  # for each neuron the overall taylor correlation
        me_score = np.full_like(lin_model_scores, np.nan)  # for each neuron the score of the 2nd order model approx.
        # for each neuron and each taylor term the correlation to the true response
        # size will depend on number of predictors in the experiment. Therefore initialize later
        taylor_by_pred: Optional[np.ndarray] = None
        all_jacobians: Optional[np.ndarray] = None
        all_hessians: Optional[np.ndarray] = None
        all_plane_abt_indices: Dict[int, List] = {}  # the cell indices for each above-threshold neuron in each plane
        all_plane_by_pred: Dict[int, List] = {}  # for each above-threshold neuron in each plane the taylor by predictor
        all_plane_true_change: Dict[int, List] = {}  # for each abt neuron in each plane the true network change
        for i, (pid, cid, tc) in enumerate(zip(plane_ids, cell_indices, test_corrs)):
            if curr_plane != pid:
                # load the data structure for this plane
                data = Data.load_direct(dfile[f"{pid}"])
                all_plane_abt_indices[pid] = []
                all_plane_by_pred[pid] = []
                all_plane_true_change[pid] = []
                curr_plane = pid
            if np.isnan(tc) or tc < c_thresh:
                continue
            predictors = data.regressor_matrix(cid)
            # compute average predictor
            tset = data.training_data(cid, 256)
            all_inputs = []
            for inp, outp in tset:
                all_inputs.append(inp.numpy())
            x_bar = np.mean(np.vstack(all_inputs), 0, keepdims=True)
            assert x_bar.shape[1] == hist_steps and x_bar.shape[2] == predictors.shape[1]
            if init_in is None:
                n_predictors = predictors.shape[1]
                init_in = np.random.randn(1, hist_steps, n_predictors).astype(np.float32)
                n_terms = (n_predictors**2 - n_predictors)//2 + n_predictors + 1  # +1 for our "behavior" pseudo-comp.
                taylor_by_pred = np.full((test_corrs.size, n_terms, 2), np.nan)
                all_jacobians = np.full((test_corrs.size, hist_steps*n_predictors), np.nan)
                all_hessians = np.full((test_corrs.size, hist_steps*n_predictors, hist_steps*n_predictors), np.nan)
                # load and initialize model
                m = model.get_standard_model(hist_steps)
            model_save_group = f"M_plane_{pid}_cell_{cid}_trained"
            w_group = model_file[model_save_group]
            weights = utilities.modelweights_from_hdf5(w_group)
            m(init_in)
            m.set_weights(weights)
            m(init_in)  # this will initialize the weights
            true_change, pred_change, by_pred = taylor_decompose(m, predictors, 5, 25)
            jacobian, hessian = d2ca_dr2(m, x_bar)
            lin_score, o2_score = complexity_scores(m, x_bar, jacobian, hessian, predictors, 5)

            # Compute prediction of Taylor expanding our network around the data-mean instead of prediction ahead as
            # above which always computes a new jacobian and hessian (the prediction here is akin to representing
            # the whole ANN with a linear model containing all first-order interaction terms):
            # Specifically, the distance to the data-mean x_bar which is used to make the prediction
            # by adding dxJ + dxHdx to model(x_bar)
            me_score[i] = o2_score
            lin_model_scores[i] = lin_score

            jacobian = jacobian.numpy().ravel()
            # reorder jacobian by n_predictor long chunks of hist_steps timeslices
            jacobian = np.reshape(jacobian, (hist_steps, n_predictors)).T.ravel()
            all_jacobians[i, :] = jacobian
            hessian = np.reshape(hessian.numpy(), (x_bar.shape[2] * hist_steps, x_bar.shape[2] * hist_steps))
            hessian = utilities.rearrange_hessian(hessian, n_predictors, hist_steps)
            all_hessians[i, :, :] = hessian
            all_plane_by_pred[pid].append(by_pred)
            all_plane_true_change[pid].append(true_change)
            taylor_corr = np.corrcoef(true_change, pred_change)[0, 1]
            taylor_full[i] = taylor_corr
            all_plane_abt_indices[pid].append(cid)
            # compute our by-predictor taylor importance as the fractional loss of r2 when excluding the component
            # add one extra "pseudo-component" which indicates the Taylor Metric when no behavior information is
            # considered (i.e. behavior-driven-neurons should have a high score here)
            off_diag_index = 0
            for row in range(n_predictors):
                for column in range(n_predictors):
                    if row == column:
                        remainder = pred_change - by_pred[:, row, column]
                        # Store in the first n_diag indices of taylor_by_pred (i.e. simply at row as indexer)
                        bsample = utilities.bootstrap_fractional_r2loss(true_change, pred_change, remainder, 1000)
                        # insert pseudo component after the first (sensory) predictor but before any detailed
                        # behavioral predictor
                        if row == 0:
                            taylor_by_pred[i, row, 0] = np.mean(bsample)
                            taylor_by_pred[i, row, 1] = np.std(bsample)
                        else:
                            taylor_by_pred[i, row+1, 0] = np.mean(bsample)
                            taylor_by_pred[i, row+1, 1] = np.std(bsample)
                    elif row < column:
                        remainder = pred_change - by_pred[:, row, column] - by_pred[:, column, row]
                        # Store in row-major order in taylor_by_pred after the first n_diag indices
                        # and after the behavior pseudo-component
                        bsample = utilities.bootstrap_fractional_r2loss(true_change, pred_change, remainder, 1000)
                        taylor_by_pred[i, n_predictors+off_diag_index+1, 0] = np.mean(bsample)
                        taylor_by_pred[i, n_predictors+off_diag_index+1, 1] = np.std(bsample)
                        off_diag_index += 1
            # compute and add the score of our behavior pseudo-component at index 1 (sensory is at index 0)
            remainder = pred_change.copy()
            for p_num in range(1, n_predictors):  # loop over all behavioral predictors (0 is sensory) and remove
                for p_num2 in range(1, n_predictors):  # together with behavior-behavior interactions
                    remainder -= by_pred[:, p_num, p_num2]
            bsample = utilities.bootstrap_fractional_r2loss(true_change, pred_change, remainder, 1000)
            taylor_by_pred[i, 1, 0] = np.mean(bsample)
            taylor_by_pred[i, 1, 1] = np.std(bsample)
        retval = int(np.sum(test_corrs >= c_thresh))
        # store everything in the hdf5 file with overwrite if requested
        create_overwrite(dfile, "linear_model_score", lin_model_scores, ovr)
        create_overwrite(dfile, "taylor_full", taylor_full, ovr)
        create_overwrite(dfile, "mean_expansion_score", me_score, ovr)
        create_overwrite(dfile, "taylor_by_pred", taylor_by_pred, ovr)
        create_overwrite(dfile, "jacobian", all_jacobians, ovr, True)
        create_overwrite(dfile, "hessian", all_hessians, ovr, True)
        # store the names of our Taylor-metrics to distinguish from the predictor names,
        # inserting also the behavior-component
        pred_names = dfile["predictor_names"][()]
        taylor_names = [pred_names[0][0]] + [np.string_("behavior")] + [pn[0] for pn in pred_names[1:]]
        taylor_names += [np.string_(f"I{int_num}") for int_num in range(taylor_by_pred.shape[1] - n_predictors - 1)]
        print(taylor_names)
        assert len(taylor_names) == taylor_by_pred.shape[1]
        create_overwrite(dfile, "taylor_names", np.vstack(taylor_names), ovr)
        for pid in all_plane_abt_indices.keys():
            d_group = dfile[f"{pid}"]
            plane_indices = all_plane_abt_indices[pid]
            plane_true_change = all_plane_true_change[pid]
            plane_by_pred = all_plane_by_pred[pid]
            if len(plane_indices) > 1:
                create_overwrite(d_group, "above_t_cell_indices", np.hstack(plane_indices), ovr, True)
                create_overwrite(d_group, "plane_true_change", np.vstack(plane_true_change), ovr, True)
                # Note: The list comprehension below is neccesary since we need to expand the dimensions of each elemet
                # in plane_by_pred so that the 3D tensors get stack along a new fourth dimension (dim0) which will
                # index the actual cell
                create_overwrite(d_group, "plane_by_pred", np.vstack([pbp[None, :] for pbp in plane_by_pred]), ovr,
                                 True)
            elif len(plane_indices) == 1:
                create_overwrite(d_group, "above_t_cell_indices", plane_indices[0], ovr)
                create_overwrite(d_group, "plane_true_change", plane_true_change[0][None, :], ovr)
                create_overwrite(d_group, "plane_by_pred", plane_by_pred[0][None, :], ovr)
    return retval


def main(d_folder: str, c_thresh: float, pool_size: int, overwrite: bool, shuffle: bool) -> None:
    """
    Runs main loop of program dividing work among processes
    :param d_folder: The main folder with fit analysis sub-folders
    :param c_thresh: The correlation threshold to use to identify a neuron as fittable
    :param pool_size: The size of the multiprocessing pool to use
    :param overwrite: If true re-analyze the experiment if it was already analyzed replacing the data
    :param shuffle: If true will perform the taylor analysis on shuffled experiments
    """
    if shuffle:
        folder_list = [f for f in os.listdir(d_folder) if path.isdir(path.join(d_folder, f)) and "_shuffle" in f
                       and ("Fish" in f or "RSpinal" in f)]
    else:
        folder_list = [f for f in os.listdir(d_folder) if path.isdir(path.join(d_folder, f)) and "_shuffle" not in f
                       and ("Fish" in f or "RSpinal" in f)]
    print(f"Processing {len(folder_list)} experiments in {folder_list} across {pool_size} processes")
    if pool_size > 1:
        pool = Pool(pool_size, maxtasksperchild=1)
        a_results = []
        for folder in folder_list:
            a_results.append(pool.apply_async(analyze_experiment, [d_folder, folder, c_thresh, overwrite]))
        for i, a in enumerate(a_results):
            processed = a.get()
            if processed >= 0:
                print(f"Analysis on experiment {i+1} out of {len(folder_list)} experiments completed.")
                print(f"Processed {processed} above-threshold units for {folder_list[i]}")
                print()
            elif processed == -1:
                print(f"Skipped analysis of {folder_list[i]} since it was already completed")
                print()
            elif processed == -2:
                print(f"Could not analyze experiment {folder_list[i]}. Failed to open fit .hdf5 file")
                print()
    else:
        # synchronous processing
        for i, folder in enumerate(folder_list):
            processed = analyze_experiment(d_folder, folder, c_thresh, overwrite)
            if processed >= 0:
                print(f"Analysis on experiment {i+1} out of {len(folder_list)} experiments completed.")
                print(f"Processed {processed} above-threshold units for {folder_list[i]}")
                print()
            elif processed == -1:
                print(f"Skipped analysis of {folder_list[i]} since it was already completed")
                print()
            elif processed == -2:
                print(f"Could not analyze experiment {folder_list[i]}. Failed to open fit .hdf5 file")
                print()


if __name__ == "__main__":
    # if the unix-default "fork" is used we cannot properly set maxtasksperchild in pool creation above
    # therefore force process creation as 'spawn' (windows and osx default)
    mp.set_start_method('spawn')

    # the following will prevent tensorflow from using the GPU - as the used models have very low complexity
    # they will generally be fit faster on the CPU - furthermore parallelization currently used
    # will not work if tensorflow is run on the GPU!!
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    a_parser = argparse.ArgumentParser(prog="rwave_decompose",
                                       description="Performs taylor decomposition and nonlinearity determination"
                                                   " on previously fit ANN models.")
    a_parser.add_argument("-f", "--folder", help="Path to folder with ANN fit subfolders", type=str, default="",
                          action=CheckArgs)
    a_parser.add_argument("-ct", "--corr_thresh", help="Threshold on test correlation to consider ANN fit for analysis",
                          type=float, default=np.sqrt(0.5))
    a_parser.add_argument("-np", "--processes", help="The number of processes across which to parallelize computation.",
                          type=int, default=1, action=CheckArgs)
    a_parser.add_argument("-ovr", "--overwrite", help="If set analyzed experiments won't be skipped but re-analyzed",
                          action='store_true')
    a_parser.add_argument("-shf", "--shuffle", help="If set shuffled fits will be analyzed", action='store_true')

    args = a_parser.parse_args()

    data_folder = args.folder
    corr_thresh = args.corr_thresh
    n_processes = args.processes
    redo = args.overwrite
    shf = args.shuffle

    main(data_folder, corr_thresh, n_processes, redo, shf)
