
"""
Script to build main file of analyzed data

Intended file information:

I Fit Neurons - only for neurons which were fit by the ann
<Neuron ID> <is lm id> <ZBrain Coords> <Barcode> <Jacobian> <Jac-cluster> <Complexity> <is vglut>

II Neuron list - where neurons come from and where they are for all neurons
<Neuron ID> <Exp-ID> <plane> <ZBrain Coords> <is ann fit> <is vglut>

III Stimuli
<Exp-ID> <plane> <temperatures>

IV Behaviors
<Exp-ID> <plane> <behaviors (separated by predictor)>

V Experiment list
<Exp-ID> <Original Filename>

VI Raw Data
<Neuron ID> <ann-testcorrs> <lm-testcorrs> <ann-shuffle-testcorrs> <lm-shuffle-testcorrs> <nl-prob>
    <mean-expansion-score> <curvatures> <NLC> <ann-trainscores> <full-taylorR2> <is ann fit>

VII Z-Brain regions
<Region ID> <Region name>

VIII Z-Brain anatomy
<Neuron ID> <1-hot region vector>
"""

import argparse
import os
from os import path
from typing import Any, List, Tuple, Dict
import h5py
import numpy as np
import utilities
import pandas as pd


def dataframe_from_hdf5(datafile: h5py.File, group_name: str, keys: List[str]) -> pd.DataFrame:
    """
    Function to generate pandas dataframe from organized hdf5 storage - this avoids using pandas builtin
    which creates hdf5 file that is not really human readable
    :param datafile: The hdf5 datafile
    :param group_name: The name of the data group to read from
    :param keys: The keys to read (columns of dataframe)
    :return: The dataframe
    """
    if group_name not in datafile:
        raise Exception(f"This datafile does not contain a group named {group_name}")
    grp = datafile[group_name]
    d = {k: [item for item in grp[k][()]] for k in keys}
    return pd.DataFrame(d)


def check_arg_sizes(*args):
    """
    Checks that all arguments have the same size along dimension 0
    :param args: The arguments to check sizes
    :return: True if all shapes along axis 0 are the same
    """
    if len(args) < 2:
        return True
    size_val = args[0].shape[0]
    for a in args[1:]:
        if a.shape[0] != size_val:
            return False
    return True


def get_fit_neurons(datafile: h5py.File) -> pd.DataFrame:
    """
    Gets information about all ann fit neurons in form of a pandas dataframe
    :param datafile: The hdf5 file from which to read the data
    :return: pandas dataframe with the fit neuron data
    """
    keys = ["neuron_id", "is_lm_fit", "zbrain_coords", "barcode", "jacobian",
            "jac_cluster", "complexity", "is_vglut", "zbrain_pixels"]
    return dataframe_from_hdf5(datafile, "fit_neurons", keys)


def save_fit_neurons(datafile: h5py.File, neuron_ids: np.ndarray, abt_lm: np.ndarray, centroids: np.ndarray,
                     barcodes: np.ndarray, jacobians: np.ndarray, jac_clusters: np.ndarray, complexities: np.ndarray,
                     is_vglut: np.ndarray):
    if not check_arg_sizes(neuron_ids, abt_lm, centroids, barcodes, jacobians, jac_clusters, complexities, is_vglut):
        raise ValueError("All inputs must have the same size along dimension 0")
    grp = datafile.create_group("fit_neurons")
    grp.create_dataset("neuron_id", data=neuron_ids)
    grp.create_dataset("is_lm_fit", data=abt_lm)
    grp.create_dataset("zbrain_coords", data=centroids)
    grp.create_dataset("barcode", data=barcodes)
    grp.create_dataset("jacobian", data=jacobians)
    grp.create_dataset("jac_cluster", data=jac_clusters)
    grp.create_dataset("complexity", data=complexities)
    grp.create_dataset("is_vglut", data=is_vglut)
    grp.create_dataset("zbrain_pixels", data=convert_h2borientum_to_zbrainpx(centroids))


def get_neuron_list(datafile: h5py.File) -> pd.DataFrame:
    """
    Gets basic information (experiment, plane, location, fit status) of all neurons in a pandas dataframe
    :param datafile: The hdf5 file from which to read the data
    :return: dataframe with the neuron list
    """
    keys = ["neuron_id", "experiment_id", "plane", "zbrain_coords", "is_ann_fit", "is_vglut", "zbrain_pixels"]
    return dataframe_from_hdf5(datafile, "neuron_list", keys)


def save_neuron_list(datafile: h5py.File, experiment_ids: np.ndarray, planes: np.ndarray, centroids: np.ndarray,
                     abt_ann: np.ndarray, is_vglut: np.ndarray):
    if not check_arg_sizes(experiment_ids, planes, centroids, abt_ann, is_vglut):
        raise ValueError("All inputs must have the same size along dimension 0")
    grp = datafile.create_group("neuron_list")
    grp.create_dataset("neuron_id", data=np.arange(experiment_ids.size))
    grp.create_dataset("experiment_id", data=experiment_ids)
    grp.create_dataset("plane", data=planes)
    grp.create_dataset("zbrain_coords", data=centroids)
    grp.create_dataset("is_ann_fit", data=abt_ann)
    grp.create_dataset("is_vglut", data=is_vglut)
    grp.create_dataset("zbrain_pixels", data=convert_h2borientum_to_zbrainpx(centroids))


def get_experiment_list(datafile: h5py.File) -> pd.DataFrame:
    keys = ["experiment_id", "experiment_name"]
    return dataframe_from_hdf5(datafile, "experiment_list", keys)


def save_experiment_list(datafile: h5py.File, experiment_ids: np.ndarray, experiment_names: np.ndarray):
    if not check_arg_sizes(experiment_ids, experiment_names):
        raise ValueError("All inputs must have the same size along dimension 0")
    grp = datafile.create_group("experiment_list")
    grp.create_dataset("experiment_id", data=experiment_ids)
    grp.create_dataset("experiment_name", data=experiment_names)


def get_raw_data(datafile: h5py.File) -> pd.DataFrame:
    keys = ["neuron_id", "ann_corr", "lm_corr", "ann_sh_corr", "lm_sh_corr", "nl_prob", "me_score", "curvature", "nlc",
            "ann_train_corr", "taylor_r2", "is_ann_fit", "is_lm_fit"]
    return dataframe_from_hdf5(datafile, "raw_data", keys)


def save_raw_data(datafile: h5py.File, ann_corrs: np.ndarray, lm_corrs: np.ndarray, ann_sh_corrs: np.ndarray,
                  lm_sh_corrs: np.ndarray, nl_probs: np.ndarray, me_scores: np.ndarray, curvatures: np.ndarray,
                  nlcs: np.ndarray, ann_train_corrs: np.ndarray, taylor_r2s: np.ndarray, abt_ann: np.ndarray,
                  abt_lm: np.ndarray):
    if not check_arg_sizes(ann_corrs, lm_corrs, ann_sh_corrs, lm_sh_corrs, nl_probs, me_scores, curvatures, nlcs,
                           ann_train_corrs, taylor_r2s, abt_ann, abt_lm):
        raise ValueError("All inputs must have the same size along dimension 0")
    grp = datafile.create_group("raw_data")
    grp.create_dataset("neuron_id", data=np.arange(ann_corrs.size))
    grp.create_dataset("ann_corr", data=ann_corrs)
    grp.create_dataset("lm_corr", data=lm_corrs)
    grp.create_dataset("ann_sh_corr", data=ann_sh_corrs)
    grp.create_dataset("lm_sh_corr", data=lm_sh_corrs)
    grp.create_dataset("nl_prob", data=nl_probs)
    grp.create_dataset("me_score", data=me_scores)
    grp.create_dataset("curvature", data=curvatures)
    grp.create_dataset("nlc", data=nlcs)
    grp.create_dataset("ann_train_corr", data=ann_train_corrs)
    grp.create_dataset("taylor_r2", data=taylor_r2s)
    grp.create_dataset("is_ann_fit", data=abt_ann)
    grp.create_dataset("is_lm_fit", data=abt_lm)


def get_stimuli(datafile: h5py.File) -> pd.DataFrame:
    keys = ["experiment_id", "plane", "temperature"]
    return dataframe_from_hdf5(datafile, "stimuli", keys)


def save_stimuli(datafile: h5py.File, experiment_ids: np.ndarray, planes: np.ndarray, temperatures: np.ndarray):
    if not check_arg_sizes(experiment_ids, planes, temperatures):
        raise ValueError("All inputs must have the same size along dimension 0")
    grp = datafile.create_group("stimuli")
    grp.create_dataset("experiment_id", data=experiment_ids)
    grp.create_dataset("plane", data=planes)
    grp.create_dataset("temperature", data=temperatures)


def get_behaviors(datafile: h5py.File) -> pd.DataFrame:
    b_names = datafile["Behavior names"][()]
    b_names = [bn[0].decode('utf-8') for bn in b_names]
    keys = ["experiment_id", "plane"] + b_names
    return dataframe_from_hdf5(datafile, "behaviors", keys)


def save_behaviors(datafile: h5py.File, experiment_ids: np.ndarray, planes: np.ndarray, behav_dict: Dict):
    if not check_arg_sizes(experiment_ids, planes, *[behav_dict[k] for k in behav_dict.keys()]):
        raise ValueError("All inputs must have the same size along dimension 0")
    grp = datafile.create_group("behaviors")
    grp.create_dataset("experiment_id", data=experiment_ids)
    grp.create_dataset("plane", data=planes)
    b_keys = list(behav_dict.keys())
    b_names = [bn.decode('utf-8') for bn in b_keys]
    for bn, bk in zip(b_names, b_keys):
        grp.create_dataset(bn, data=behav_dict[bk])


def get_cadata(datafile: h5py.File) -> pd.DataFrame:
    keys = ["experiment_id", "plane", "neuron_id", "ca_data"]
    return dataframe_from_hdf5(datafile, "activity", keys)


def save_cadata(datafile: h5py.File, experiment_ids: np.ndarray, planes: np.ndarray, neuron_ids: np.ndarray,
                cadata: np.ndarray):
    if not check_arg_sizes(experiment_ids, planes, neuron_ids, cadata):
        raise ValueError("All inputs must have the same size along dimension 0")
    grp = datafile.create_group("activity")
    grp.create_dataset("experiment_id", data=experiment_ids)
    grp.create_dataset("plane", data=planes)
    grp.create_dataset("neuron_id", data=neuron_ids)
    grp.create_dataset("ca_data", data=cadata, compression="gzip", compression_opts=9)


def get_zbrain_region_list(datafile: h5py.File) -> pd.DataFrame:
    keys = ["region_index", "region_name"]
    return dataframe_from_hdf5(datafile, "zbrain_region_list", keys)


def save_zbrain_region_list(datafile: h5py.File, region_names: List[str]):
    region_ids = np.arange(len(region_names))
    rnl = [np.string_(rn) for rn in region_names]
    grp = datafile.create_group("zbrain_region_list")
    grp.create_dataset("region_index", data=region_ids)
    grp.create_dataset("region_name", data=np.vstack(rnl))


def get_zbrain_anatomy(datafile: h5py.File) -> pd.DataFrame:
    keys = ["neuron_id", "region_membership"]
    return dataframe_from_hdf5(datafile, "zbrain_anatomy", keys)


def save_zbrain_anatomy(datafile: h5py.File, neuron_ids: np.ndarray, region_memberships: np.ndarray):
    if not check_arg_sizes(neuron_ids, region_memberships):
        raise ValueError("All inputs must have the same size along dimension 0")
    grp = datafile.create_group("zbrain_anatomy")
    grp.create_dataset("neuron_id", data=neuron_ids)
    grp.create_dataset("region_membership", data=region_memberships)


def convert_h2borientum_to_zbrainpx(cent_um: np.ndarray) -> np.ndarray:
    """
    Takes um coordinates that have been transformed to the z-brain but are within the H2B brain coordinate axes
    (x=L->R; y=A->P; z=D->V) and converts them to pixel coordinates in the z-brain axes space (x=A->P; y=R->L; z=V->D)
    :param cent_um: Nx3 array of um coordinates
    :return: Nx3 array of pixel coordinates
    """
    x_pix_h2b = np.round((cent_um[:, 0] / 0.798)[:, None])
    y_pix_h2b = np.round((cent_um[:, 1] / 0.798)[:, None])
    z_pix_h2b = np.round((cent_um[:, 2] / 2)[:, None])
    z_pix_zb = 138 - z_pix_h2b  # z-brain has 138 slices
    y_pix_zb = x_pix_h2b
    x_pix_zb = 1406 - y_pix_h2b
    return np.c_[x_pix_zb, y_pix_zb, z_pix_zb].astype(int)


def regionsort(filename: str) -> int:
    """
    Returns a sort key that first sorts according to brain region and then according to experiment number
    """
    k = int(filename[4:6])  # the fish number
    if "FBd" in filename:
        return k + 100
    elif "FBv" in filename:
        return k + 200
    elif "MBd" in filename:
        return k + 300
    elif "MBv" in filename:
        return k + 400
    elif "HBd" in filename:
        return k + 500
    elif "HBv" in filename:
        return k + 600
    else:
        raise ValueError(f"Filename {filename} has no recognizable imaging region")


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


def extract_stimuli(base_folder: str, e_folders: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts stimuli from experiments concatenating all experiments and planes
    :param base_folder: The base folder with the analysis directories and experiment hdf5 files
    :param e_folders: The experiment analysis folders from which to extract stimuli
    :return:
        [0]: The temperature stimuli
        [1]: The corresponding experimental planes
        [2]: The corresponding experiment ids
    """
    stimuli = []
    planes = []
    exp_ids = []

    for i, e_folder in enumerate(e_folders):
        datafile_path = path.join(base_folder, path.join(e_folder, f"{e_folder}_ANN_analysis.hdf5"))
        with h5py.File(datafile_path, 'r') as dfile:
            n_planes = int(dfile["n_planes"][()])
            planes.append(np.arange(n_planes))
            exp_ids.append(np.full(n_planes, i))
            for pid in range(n_planes):
                stimulus = dfile[f"{pid}"]["raw_stimulus"][()]
                stimuli.append(stimulus)
    # fix length discrepancies by trimming
    shortest = min([s.size for s in stimuli])
    stimuli = [s[:shortest] for s in stimuli]
    return np.vstack(stimuli), np.hstack(planes), np.hstack(exp_ids)


def extract_behaviors(base_folder: str, e_folders: List[str], b_names: np.ndarray) -> Tuple[Dict, np.ndarray,
                                                                                            np.ndarray]:
    """
    Extracts named behaviors from experiments concatenating across experiments and planes
    :param base_folder: The base folder with the analysis directories and experiment hdf5 files
    :param e_folders: The experiment analysis folders from which to extract behaviors
    :param b_names: The predictor names of the behaviors
    :return:
        [0]: Dictionary with behavior predictor names as keys and all corresponding traces as ndarray
        [1]: The experimental planes
        [2]: The experiment ids
    """
    behaviors = []
    planes = []
    exp_ids = []

    for i, e_folder in enumerate(e_folders):
        datafile_path = path.join(base_folder, path.join(e_folder, f"{e_folder}_ANN_analysis.hdf5"))
        with h5py.File(datafile_path, 'r') as dfile:
            n_planes = int(dfile["n_planes"][()])
            planes.append(np.arange(n_planes))
            exp_ids.append(np.full(n_planes, i))
            for pid in range(n_planes):
                behavior = dfile[f"{pid}"]["raw_behavior"][()]
                behaviors.append(behavior)
    # fix length discrepancies by trimming
    shortest = min([b.shape[1] for b in behaviors])
    behaviors = [b[:, :shortest] for b in behaviors]
    # convert from n_behaviors x n_timepoints array into dictionary grouping according to behavior name
    b_dict = {k[0]: np.vstack([bhv[i, :] for bhv in behaviors]) for i, k in enumerate(b_names)}
    return b_dict, np.hstack(planes), np.hstack(exp_ids)


def get_calcium_data(base_folder: str, e_folders: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns calcium data across all planes and experiments
    :param base_folder: The folder with the experiment hdf5 files
    :param e_folders: List of experiment folders
    :return:
        [0]: Calcium data used for fitting
        [1]: Planes
        [2]: Experiment IDs
        [3]: Neuron IDs
    """
    ca_data = []
    planes = []
    exp_ids = []

    for i, e_folder in enumerate(e_folders):
        datafile_path = path.join(base_folder, path.join(e_folder, f"{e_folder}_ANN_analysis.hdf5"))
        with h5py.File(datafile_path, 'r') as dfile:
            n_planes = int(dfile["n_planes"][()])
            for pid in range(n_planes):
                ca = dfile[f"{pid}"]["ca_responses"][()]
                ca_data.append(ca)
                planes.append(np.full(ca.shape[0], pid))
                exp_ids.append(np.full(ca.shape[0], i))
    # fix length discrepancies by trimming
    shortest = min([c.shape[1] for c in ca_data])
    ca_data = [c[:, :shortest] for c in ca_data]
    ca_data = np.vstack(ca_data)
    planes = np.hstack(planes)
    exp_ids = np.hstack(exp_ids)
    return ca_data, planes, exp_ids, np.arange(planes.size)


def load_zbrain_region_names(zbr_file_path: str) -> List[str]:
    """
    Load zbrain region names from file
    :param zbr_file_path: Path to the zbrain region hdf5 file
    :return:
        [0]: List of region names
    """
    with h5py.File(zbr_file_path, 'r') as zbr_file:
        fills = zbr_file["Fills"]
        return sorted([k for k in fills.keys()])


def _zcoord_invalid(coord: np.ndarray) -> bool:
    """
    Checks if a coordinate falls within the z-brain space
    :param coord: 3-long vector of x, y, z coordinates
    :return: True if the coordinate is invalid
    """
    if coord[0] < 0 or coord[0] >= 1406:
        return True
    if coord[1] < 0 or coord[1] >= 612:
        return True
    if coord[2] < 0 or coord[2] >= 138:
        return True
    return False


def build_zbrain_region_barcodes(cent_px: np.ndarray, region_names: List[str], zbr_file_path: str,
                                 fast: bool) -> np.ndarray:
    """
    For each pixel centroid get a barcode of which zbrain regions it belongs to
    :param cent_px: n_centroids x 3 array of z-brain pixel coordinates
    :param region_names: The region_names to check in the datafile
    :param zbr_file_path: Path to the zbrain region hdf5 file
    :param fast: If set, all masks will be loaded into memory (~32GB) otherwise processing will happen on file level
    :return: n x len(region_names) array of 0/1 barcodes indicating whether a centroid lies within a region (1) or not
    """
    if cent_px.ndim != 2 or cent_px.shape[1] != 3:
        raise ValueError(f"cent_px has to be n_points x 3 array of pixel coordinates but has shape {cent_px.shape}")
    cent_px = cent_px.astype(int)
    rval = np.zeros((cent_px.shape[0], len(region_names)), dtype=bool)
    if not fast:
        with h5py.File(zbr_file_path, 'r') as zbr_file:
            fills = zbr_file["Fills"]
            for i, c in enumerate(cent_px):
                if _zcoord_invalid(c):
                    continue
                for j, name in enumerate(region_names):
                    if fills[name][c[0], c[1], c[2]]:
                        rval[i, j] = True
    else:
        masks = np.empty((len(region_names), 1406, 621, 138), dtype=bool)
        with h5py.File(zbr_file_path, 'r') as zbr_file:
            fills = zbr_file["Fills"]
            for mi, name in enumerate(region_names):
                masks[mi, :, :, :] = fills[name][()]
        for i, c in enumerate(cent_px):
            if _zcoord_invalid(c):
                continue
            rval[i, :] = masks[:, c[0], c[1], c[2]]
    return rval


def get_data(base_folder: str, e_folder: str, eid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                                 np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                                 np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                                 np.ndarray]:
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
        [4]: n x 3 matrix of unit reference brain centroids
        [5]: n long vector of nlc metric scores
        [6]: n long vector of glutamate channel brightness
        [7]: n long vector of linear model test correlations
        [8]: n x (n_timepointsxn_predictors) matrix of all jacobians at the data mean => linear receptive fields
        [9]: n long vector of data mean expansion scores
        [10]: n long vector of the planes each neuron came frome
        [11]: n long vector of ann training correlations
        [12]: n long vector of full taylor R2
    """
    datafile_path = path.join(base_folder, path.join(e_folder, f"{e_folder}_ANN_analysis.hdf5"))
    centroidfile_path = path.join(base_folder, path.join("CellCoordinates",
                                                         f"{e_folder.replace('_shuffle','')}_transformed_zbrain.csv"))
    glut_channel = []
    plane_ids = []
    with h5py.File(datafile_path, 'r') as dfile:
        jacobian = dfile["jacobian"][()]
        test_corrs = dfile["correlations_test"][()]
        train_corrs = dfile["correlations_trained"][()]
        lm_test_corrs = dfile["lm_correlations_test"][()]
        taylor_scores = dfile["taylor_by_pred"][()]
        curve_metrics = dfile["avg_curvatures"][()]
        nlc_metrics = dfile["nlc_metrics"][()]
        n_planes = dfile["n_planes"][()]
        me_scores = dfile["mean_expansion_score"][()]**2  # immediately store as R2 not correlation coefficient
        taylor_r2 = dfile["taylor_full"][()]**2
        for p_id in range(n_planes):
            plane_group = dfile[f"{p_id}"]
            vg_plane = plane_group["anatomy_brightness"][()]
            plane_ids.append(np.full(vg_plane.size, p_id))
            glut_channel.append(vg_plane)
    glut_channel = np.hstack(glut_channel)
    plane_ids = np.hstack(plane_ids)
    assert glut_channel.shape[0] == test_corrs.size
    if test_corrs.size != taylor_scores.shape[0] or test_corrs.size != curve_metrics.size:
        raise ValueError(f"Analyzed metrics in experiment {e_folder} have mismatching sizes")
    exp_ids = np.full(curve_metrics.size, eid)
    centroids = np.genfromtxt(centroidfile_path, delimiter=',', skip_header=True)
    cent_nan = np.full(4, np.nan)
    centroids[centroids[:, 3] == 1, :] = cent_nan
    centroids = centroids[:, :3].copy()
    return test_corrs, taylor_scores, curve_metrics, exp_ids, centroids, nlc_metrics, glut_channel, lm_test_corrs,\
        jacobian, me_scores, plane_ids, train_corrs, taylor_r2


def get_data_sh(base_folder: str, e_folder: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get relevant information to analyze from shuffle experiment
    :param base_folder: The base folder with all experiment hdf5 files
    :param e_folder: The analysis folder of the experiment in question
    :return:
        [0]: The test correlations of all units in the experiment
        [1]: The test correlations of the linear model for all units in the experiment
    """
    datafile_path = path.join(base_folder, path.join(e_folder, f"{e_folder}_ANN_analysis.hdf5"))
    with h5py.File(datafile_path, 'r') as dfile:
        test_corrs = dfile["correlations_test"][()]
        lm_test_corrs = dfile["lm_correlations_test"][()]
    return test_corrs, lm_test_corrs


def main(folder: str, c_thresh: float, sig_thresh: float, nonlin_thresh: float, lm_thresh: float, me_thresh: float,
         save_activity: bool, outfile: str):
    """
    Runs analysis
    :param folder: The folder with experiment hdf5 files and ann analysis subfolders
    :param c_thresh: The correlation threshold above which to consider units
    :param sig_thresh: The taylor metric threshold - metrics not significantly above this will be set to 0
    :param nonlin_thresh: The threshold for considering a neuron nonlinear
    :param lm_thresh: The threshold for considering a neuron identified by the linear comparison model
    :param me_thresh: The threshold on the mean expansion R2 to consider the neuron describable by a 2nd order model
    :param save_activity: If true, calcium data for all neurons will be saved to the file as well
    :param outfile: The name of the output file
    """
    experiment_ids, experiment_names = [], []  # to relate experiment ids with their respective names
    # get list of ANN analysis subfolders
    folder_list = os.listdir(folder)
    sub_folder_list_sh = [f for f in folder_list if path.isdir(path.join(folder, f)) and "Fish" in f
                          and "_shuffle" in f]
    sub_folder_list_sh.sort(key=regionsort)  # sort according to brain region
    sub_folder_list = [f for f in folder_list if path.isdir(path.join(folder, f)) and "Fish" in f
                       and "_shuffle" not in f]
    sub_folder_list.sort(key=regionsort)  # sort according to brain region
    # Create dictionary (and then dataframe) to save per-experiment information
    # for our overview sheet
    info_dict = {"Experiment name": [], "N Caiman": [], "N Fit": [], "N Cent. mapped": []}
    # the ann test trial correlations
    all_test_corrs = []
    # the linear comparison model test trial correlations
    all_lm_test_corrs = []
    # the ann shuffle test correlations
    all_test_corrs_sh = []
    # the linear model shuffle test correlations
    all_lm_tc_sh = []
    # the raw taylor scores which include bootstrap information
    all_taylor_scores_raw = []
    # the curvature metrics
    all_curve_metrics = []
    # the experiment ids for each neuron
    all_exp_ids = []
    # the plane that each neuron came from
    all_plane_ids = []
    # the centroids of each neuron
    all_centroids = []
    # the nlc values
    all_nlc_metrics = []
    # the vglut channel brightness of each neuron
    all_vglut = []
    # the jacobians at the data mean (=receptive field) of each neuron
    all_jacobians = []
    # the data mean expansion R2 for each neuron
    all_mescores = []
    # the ann training trial correlations
    all_ann_train_corrs = []
    # the full taylor r2
    all_taylor_r2 = []
    for i, flder in enumerate(sub_folder_list):
        info_dict["Experiment name"].append(flder)
        experiment_ids.append(i)
        experiment_names.append(flder.encode())
        tc, ts, cm, ei, cn, nlc, vg, lmtc, jacs, mescores, pids, trainc, taylorr2 = get_data(folder, flder, i)
        info_dict["N Caiman"].append(np.sum(np.isfinite(tc)))
        info_dict["N Fit"].append(np.sum(tc >= c_thresh))
        info_dict["N Cent. mapped"].append(np.sum(np.logical_and(tc >= c_thresh, np.isfinite(cn[:, 0]))))
        tc_sh, lm_tc_sh = get_data_sh(folder, sub_folder_list_sh[i])
        all_test_corrs.append(tc)
        all_lm_test_corrs.append(lmtc)
        all_test_corrs_sh.append(tc_sh)
        all_lm_tc_sh.append(lm_tc_sh)
        all_taylor_scores_raw.append(ts)
        all_curve_metrics.append(cm)
        all_exp_ids.append(ei)
        all_plane_ids.append(pids)
        all_centroids.append(cn)
        all_nlc_metrics.append(nlc)
        all_vglut.append(vg)
        all_jacobians.append(jacs)
        all_mescores.append(mescores)
        all_ann_train_corrs.append(trainc)
        all_taylor_r2.append(taylorr2)

    info_dframe = pd.DataFrame(info_dict)
    info_dframe.to_csv(path.join(folder, f"Overview_Info_{path.splitext(outfile)[0]}.csv"))

    # convert lists into appropriate arrays
    all_test_corrs = np.hstack(all_test_corrs)
    all_ann_train_corrs = np.hstack(all_ann_train_corrs)
    all_taylor_r2 = np.hstack(all_taylor_r2)
    all_lm_test_corrs = np.hstack(all_lm_test_corrs)
    all_lm_tc_sh = np.hstack(all_lm_tc_sh)
    all_test_corrs_sh = np.hstack(all_test_corrs_sh)
    all_vglut = np.hstack(all_vglut)
    all_taylor_scores_raw = np.vstack(all_taylor_scores_raw)
    all_centroids = np.vstack(all_centroids)
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
    all_jacobians = np.vstack(all_jacobians)
    all_mescores = np.hstack(all_mescores)

    above_thresh_ann = all_test_corrs >= corr_th
    above_thresh_lm = all_lm_test_corrs >= lm_thresh
    ann_fit_is_lm_fit = above_thresh_lm[above_thresh_ann]
    ann_fit_neuron_ids = np.arange(all_test_corrs.size)[above_thresh_ann]

    # restrict jacobians to fit neurons
    all_jacobians = all_jacobians[above_thresh_ann, :]

    # Based on images the threshold to ID a vglut-positive neuron should be 1<=T<=2. 1 will include a few doubtful
    # neurons but 2 will exclude some fairly clear positives.
    vg_thresh = 1.5
    glutamatergic = all_vglut > vg_thresh

    # compute nonlinearity probabilities
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

    # compute complexity score on fit neurons (0=linear, 1=nonlinear 2nd order model works, 2=2nd order model fails)
    complexity = np.zeros(np.sum(above_thresh_ann), dtype=int)
    complexity[np.logical_and(is_nonlinear, all_mescores[above_thresh_ann] >= me_thresh)] = 1
    complexity[np.logical_and(is_nonlinear, all_mescores[above_thresh_ann] < me_thresh)] = 2

    # perform jacobian (receptive field) based subclustering on sensory neurons - -1 will mean sensory but not
    # subclustered while -2 will identify neurons for which no clustering has been performed
    all_fit_jac_cluster_ids = np.full(np.sum(above_thresh_ann), -2, dtype=int)
    # find types of interest according to cluster
    nl_id = labels.index(b"Nonlin")
    t_id = labels.index(b"Temperature")
    # behavior is from the "behavior" catch all until the first interaction term
    any_b_id = slice(labels.index(b"behavior"), [loc for loc, s in enumerate(labels) if b"I" in s][0])
    # interactors are at the end
    any_interact = slice([loc for loc, s in enumerate(labels) if b"I" in s][0], barcode.shape[1])
    non_linear = barcode[:, nl_id] == 1
    linear = np.logical_not(non_linear)
    sens_contrib = barcode[:, t_id] == 1
    mot_contrib = np.sum(barcode[:, any_b_id] == 1, 1) >= 1
    int_contrib = np.sum(barcode[:, any_interact], 1) >= 1
    non_sens_contrib = np.logical_or(mot_contrib, int_contrib)
    sensory_lin_cluster = np.logical_and(np.logical_and(linear, sens_contrib), np.logical_not(non_sens_contrib))
    sensory_nl_cluster = np.logical_and(np.logical_and(non_linear, sens_contrib), np.logical_not(non_sens_contrib))
    sensory_cluster = np.logical_or(sensory_nl_cluster, sensory_lin_cluster)  # pure sensory
    # get slice indexers for hessian/jacobian based on predictor labels
    assert labels[0] == b"Nonlin"  # further logic is based on the assumption that Nonlin is the first label
    slice_sensory = slice((t_id - 1) * 50, t_id * 50)  # each predictor has 50 timepoints
    # combine all sensory jacobians
    jac_sensory = all_jacobians[sensory_cluster, :][:, slice_sensory]
    # cluster all sensory (linear and nonlinear) units according to the jacobian
    j_cluster_min_size = 25
    j_cluster_min_cosim = 0.8
    m_ship_jac = utilities.greedy_cosine_cluster(jac_sensory, j_cluster_min_size, j_cluster_min_cosim)
    all_fit_jac_cluster_ids[sensory_cluster] = m_ship_jac

    # extract names of behavior predictors from labels
    with h5py.File(datafile_path, 'r') as dfile:
        predictor_names = dfile["predictor_names"][()]
    behavior_names = predictor_names[1:]

    with h5py.File(os.path.join(folder, outfile), "w") as main_file:
        main_file.create_dataset("ANN test correlation threshold", data=c_thresh)
        main_file.create_dataset("Taylor threshold", data=sig_thresh)
        main_file.create_dataset("Nonlinearity threshold", data=nonlin_thresh)
        main_file.create_dataset("Linear model threshold", data=lm_thresh)
        main_file.create_dataset("Mean expansion R2 threshold", data=me_thresh)
        main_file.create_dataset("Taylor corrected significance", data=(1-min_significance))
        main_file.create_dataset("Taylor n_sigma to reach significance", data=n_sigma)
        main_file.create_dataset("Barcode labels", data=np.vstack(labels))
        main_file.create_dataset("Minimal jacobian cluster size", data=j_cluster_min_size)
        main_file.create_dataset("Minimal jacobian cluster cosine similarity",  data=j_cluster_min_cosim)
        main_file.create_dataset("Behavior names", data=behavior_names)
        save_neuron_list(main_file, all_exp_ids, all_plane_ids, all_centroids, above_thresh_ann, glutamatergic)
        save_experiment_list(main_file, np.hstack(experiment_ids), np.hstack(experiment_names))
        save_raw_data(main_file, all_test_corrs, all_lm_test_corrs, all_test_corrs_sh, all_lm_tc_sh, nonl_probabilities,
                      all_mescores, all_curve_metrics, all_nlc_metrics, all_ann_train_corrs, all_taylor_r2,
                      above_thresh_ann, above_thresh_lm)
        save_fit_neurons(main_file, ann_fit_neuron_ids, ann_fit_is_lm_fit, all_centroids[above_thresh_ann, :], barcode,
                         all_jacobians, all_fit_jac_cluster_ids, complexity, glutamatergic[above_thresh_ann])
        s, p, e = extract_stimuli(folder, sub_folder_list)
        save_stimuli(main_file, e, p, s)
        b, p, e = extract_behaviors(folder, sub_folder_list, behavior_names)
        save_behaviors(main_file, e, p, b)
        # obtain zbrain anatomy
        region_names = load_zbrain_region_names("ZBrain_Masks.hdf5")
        save_zbrain_region_list(main_file, region_names)
        region_memberships = build_zbrain_region_barcodes(convert_h2borientum_to_zbrainpx(all_centroids), region_names,
                                                          "ZBrain_Masks.hdf5", True)
        save_zbrain_anatomy(main_file, np.arange(region_memberships.shape[0]), region_memberships)
        if save_activity:
            c, p, e, n = get_calcium_data(folder, sub_folder_list)
            valid = np.isfinite(all_test_corrs)
            save_cadata(main_file, e[valid], p[valid], n[valid], c[valid, :])


if __name__ == "__main__":
    a_parser = argparse.ArgumentParser(prog="rwave_build_main",
                                       description="Will create main file of analysis results for all figure "
                                                   "plotting")
    a_parser.add_argument("-f", "--folder", help="Path to folder with experiment hdf5 files", type=str, default="",
                          action=CheckArgs)
    a_parser.add_argument("-co", "--corrthresh", help="The fit test correlation threshold", type=float,
                          default=np.sqrt(0.5))
    a_parser.add_argument("-si", "--sigthresh", help="The threshold for taylor metric significance", type=float,
                          default=0.1)
    a_parser.add_argument("-nl", "--nlthresh", help="The threshold of nonlinearity", type=float, default=0.5)
    a_parser.add_argument("-lm", "--lmthresh", help="The threshold of lm fit test correlation", type=float,
                          default=np.sqrt(0.5))
    a_parser.add_argument("-me", "--methresh", help="The threshold of the R2 for considering neuron fit by 2nd order "
                                                    "model around the data mean.", type=float, default=0.5)
    a_parser.add_argument("-svc", "--save_ca", help="If set calcium data will be saved", action='store_true')
    a_parser.add_argument("-o", "--output", help="The name of the output file", default="main_analysis.hdf5")

    cl_args = a_parser.parse_args()

    data_folder = cl_args.folder
    corr_th = cl_args.corrthresh
    sig_th = cl_args.sigthresh
    nl_th = cl_args.nlthresh
    lm_th = cl_args.lmthresh
    me_th = cl_args.methresh
    svc = cl_args.save_ca
    ofile = cl_args.output

    main(data_folder, corr_th, sig_th, nl_th, lm_th, me_th, svc, ofile)
