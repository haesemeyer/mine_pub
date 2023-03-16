"""
Script to organize and process data from Musall et al., 2019
"""
import os
from os import path
import h5py
import numpy as np
from scipy.io import loadmat
from typing import List, Tuple, Any, Optional
import argparse
import multiprocessing as mp
from multiprocessing import Pool
from mine import Mine, MineData


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
        else:
            raise Exception("Parser was asked to check unknown argument")


def extract_predictor_labels(org_regdata_mat: h5py.File) -> List[str]:
    """
    Parse object references to re-create proper strings from MATLAB encoding of recLabels datafield
    :param org_regdata_mat: The orgRegData.mat file
    :return: List of all predictor labels (recLabels)
    """
    return [u''.join(chr(c[0]) for c in org_regdata_mat[ref[0]]) for ref in org_regdata_mat['recLabels']]


def collate_unique_predictors(org_regdata_mat: h5py.File) -> Tuple[np.ndarray, List[str]]:
    """
    Creates an n_predictors x n_timepoints matrix of unique (non-timeshifted) predictors excluding video
    :param org_regdata_mat: The orgregData.mat file
    :return:
        [0]: n_predictors x n_timepoints matrix of predictors
        [1]: n_predictors long list of corresponding labels
    """
    # Not all predictors are used/present for all sessions. We therefore need to find the existing ones from
    # all *possible* ones
    # get all *possible* predictor labels
    all_labels = extract_predictor_labels(org_regdata_mat)
    # this indicator is 1 for all skipped *possible* predictor columns
    idx = org_regdata_mat["idx"][()]
    # this variable uses MATLAB 1-based indexing to index from all *possible* predictor columns into the labels
    rec_idx = org_regdata_mat["recIdx"][()]
    # this variable contains the full set of possibly time-shifted predictors
    full_r = org_regdata_mat["fullR"]
    ret_dict = {}
    # ensure that we can identify the labels for rows in full_r
    assert rec_idx.size - np.sum(idx) == full_r.shape[0]
    # every entry in used_idx contains the predictor label index for each row in full_r
    used_idx = rec_idx[idx == 0].astype(int) - 1  # convert to 0-based index
    # NOTE: We need to take care of the nature of the shifts used in the fullR design matrix
    # For all task variables [time, Choice, reward, lVisStim, rVisStim, lAudStim, rAudStim, prevReward, prevChoice,
    #   prevMod, water] the first index will give us the 0-shift predictor
    # For all variables that contain analog (as well as kernel) responses [piezo, whisk, nose, fastPupil, slowPupil,
    #   face, body] the first index will give us the analog predictor which we want to use
    # For all kernel-only movement variables [lGrab, rGrab, lLick, rLick] however, we need to add 15 to the index
    #   according to (https://github.com/churchlandlab/ridgeModel/blob/master/tutorial_linearModel.m) to obtain
    #   the 0-shift predictor
    move_kernel_names = ["lGrab", "rGrab", "lLick", "rLick"]
    for row, label_ix in enumerate(used_idx):
        name = all_labels[label_ix]
        if name == 'Move' or name == 'bhvVideo':
            continue
        if name in ret_dict:
            continue
        if name in move_kernel_names:
            ret_dict[name] = full_r[row+15]
        else:
            ret_dict[name] = full_r[row]
    used_labels = list(ret_dict.keys())
    return np.vstack([ret_dict[k] for k in used_labels]), used_labels


def load_response_data(interp_vc_mat: str) -> np.ndarray:
    """
    Extracts interpolated and trial concatenated temporal components from interpVC.mat
    :param interp_vc_mat: The filename and path to the interpVc.mat (oldstyle matlab) file
    :return: n_temporal_components (200) x n_timepoints matrix of widefield data components
    """
    file_data = loadmat(interp_vc_mat)
    return file_data["Vc"]


def get_session_data(session_folder: str) -> Tuple[List[np.ndarray], np.ndarray, List[str], np.ndarray]:
    """
    Creates a train/test data object from the predictor and response data of one session
    :param session_folder: The path to the folder of the session
    :return:
        [0]: List of individual n_timepoints long predictors that apply to the entire session
        [1]: n_components (200) x n_timepoints matrix of responses
        [2]: The names of the predictors
        [3]: n_components (200) x width x height array of spatial components
    """
    f_list = os.listdir(session_folder)
    if "orgregData.mat" not in f_list or "interpVc.mat" not in f_list or "Vc.mat" not in f_list:
        raise IOError(f"Not all required datafiles are in folder {session_folder}. Found {f_list}.")
    responses = load_response_data(path.join(session_folder, "interpVc.mat"))
    with h5py.File(path.join(session_folder, "orgregData.mat"), 'r') as dfile:
        pred_mat, pred_labels = collate_unique_predictors(dfile)
    with h5py.File(path.join(session_folder, "Vc.mat"), 'r') as dfile:
        spatial = dfile['U'][()]
    predictors = [p for p in pred_mat]
    return predictors, responses, pred_labels, spatial


def analyze_data(d_folder, s_folder) -> Tuple[Any, Any, Any, Any, Optional[MineData]]:
    """
    Processes all data of a given session
    :param d_folder: The root datafolder
    :param s_folder: The name of the session
    :return:
        [0]: List of individual n_timepoints long predictors that apply to the entire session
        [1]: n_components (200) x n_timepoints matrix of responses
        [2]: The names of the predictors
        [3]: The spatial components n_components x width x height
        [3]: mine_data object containing the analysis output
    """
    print(f"Processing {s_folder}")
    try:
        predictors, responses, predictor_labels, spatial = get_session_data(path.join(d_folder, s_folder))
        # Relative datashift to virtually re-center convolutional filters - we set the current time
        # in the response to a future timepoint in the predictors, half a filter length ahead.
        # That way the convolutional filters will be centered around the current time rather than point
        # exclusively into the past
        re_center = 75
        r = responses[:, :-re_center].copy()
        p = [pred[re_center:].copy() for pred in predictors]
        # now the point re_center in predictors will be aligned with 0 in respones
    except IOError:
        print(f"{s_folder} does not seem to be a folder with session data. Skipping")
        return None, None, None, None, None
    print("Data loaded")
    # standardize predictors and responses to 0 mean and 1 standard deviation
    for pred in p:
        pred -= np.mean(pred)
        pred /= (np.std(pred) + 1e-16)
    r -= np.mean(r, 1, keepdims=True)
    r /= np.std(r, 1, keepdims=True)
    with h5py.File(path.join(d_folder, f"{s_folder}_models.hdf5"), 'w') as model_file:
        # corr-cut set to 0.1 to only fully process data with at least 1% explained variance on test
        miner = Mine(2 / 3, 150, 0.1, True, True, 30, 143)
        miner.model_weight_store = model_file
        mdata = miner.analyze_data(p, r)

    return predictors, responses, predictor_labels, spatial, mdata


def save_data(session_name: str, hdf: h5py.File, predictors: List, predictor_labels: List, responses: np.ndarray,
              spatial: np.ndarray, mdata: MineData) -> None:
    """
    Save input and output data to hdf5
    :param session_name: The name of the session to save
    :param hdf: The hdf5 file object
    :param predictors: List of the predictors in the session
    :param predictor_labels: The names of the predictors
    :param responses: All component responses of this session
    :param spatial: The spatial components from this session
    :param mdata: The Miner output data object
    """
    s_group = hdf.create_group(session_name)
    i_group = s_group.create_group("input_data")
    i_group.create_dataset(name="predictor_labels", data=np.vstack([np.string_(s) for s in predictor_labels]))
    i_group.create_dataset(name="responses", data=responses)
    i_group.create_dataset(name="predictors", data=np.vstack(predictors))
    i_group.create_dataset(name="spatial", data=spatial)
    # save output data
    o_group = s_group.create_group("output_data")
    mdata.save_to_hdf5(o_group, True)


def main(data_folder: str, out_file_name: str, n_processes: int) -> None:
    """
    Main program loop
    :param data_folder: The folder with all session-sub-folders
    :param out_file_name: The name of the output file we generate
    :param n_processes: The number of independent processes to use
    """
    session_folders = os.listdir(data_folder)
    # filter out files and only keep folders
    session_folders = [sf for sf in session_folders if path.isdir(path.join(data_folder, sf))]

    all_async_results = []
    if n_processes > 1:
        pool = Pool(n_processes, maxtasksperchild=1)
    else:
        pool = None

    with h5py.File(path.join(data_folder, out_file_name), 'w') as outfile:
        for sf in session_folders:
            if pool is None:
                # process synchronously
                preds, res, plabels, sptl, mine_data = analyze_data(data_folder, sf)
                # preds: n_predictors long list of n_timepoints long vectors of predictors
                # res: n_components x n_timepoints matrix of temporal widefield response components
                # plabels: n_predictors long list of predictor names
                # sptl: n_components x width x height array of spatial widefield response components
                if preds is None:
                    # occurs upon errors of data loading
                    continue
                save_data(sf, outfile, preds, plabels, res, sptl, mine_data)
            else:
                ares = pool.apply_async(analyze_data, [data_folder, sf])
                all_async_results.append((ares, sf))
        if pool is not None:
            for ar in all_async_results:
                preds, res, plabels, sptl, mine_data = ar[0].get()
                if preds is not None:
                    save_data(ar[1], outfile, preds, plabels, res, sptl, mine_data)


if __name__ == "__main__":

    # if the unix-default "fork" is used we cannot properly set maxtasksperchild in pool creation above
    # therefore force process creation as 'spawn' (windows and osx default)
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        # If the script is called twice, context cannot be set again
        pass

    # the following will prevent tensorflow from using the GPU - as the used models have very low complexity
    # they will generally be fit faster on the CPU - furthermore parallelization currently used
    # will not work if tensorflow is run on the GPU!!
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    a_parser = argparse.ArgumentParser(prog="processMusall",
                                       description="Applies MINE to part of the data from Musall et al., NatNeuro,"
                                                   " 2019")
    a_parser.add_argument("-f", "--folder", help="Path to folder with session subfolders", type=str, default="",
                          action=CheckArgs)
    a_parser.add_argument("-o", "--outname", help="Name of the output hdf5 file to generate", type=str,
                          default="musall.hdf5")
    a_parser.add_argument("-np", "--processes", help="The number of processes across which to parallelize computation.",
                          type=int, default=1, action=CheckArgs)

    args = a_parser.parse_args()

    dfolder = args.folder
    out_name = args.outname
    n_proc = args.processes

    main(dfolder, out_name, n_proc)
