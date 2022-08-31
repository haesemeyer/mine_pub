
"""
Script for identifying stimulus and behavior related neurons in random-wave experiment data
"""

import model
import numpy as np
import utilities
import h5py
from scipy.ndimage import gaussian_filter1d
import os
from os import path
from experiment import Experiment2P
from numba import njit
from multiprocessing import Pool
import multiprocessing as mp
from typing import Union, Any, List, Tuple, Optional
import argparse
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from collections import Counter


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


def convert_laser_current(input_current: np.ndarray) -> np.ndarray:
    """
    Based on power measurements converts the current driving the laser
    into the power at sample
    :param input_current: Currents in mA
    :return: Power in mW
    """
    pas = input_current * 0.17683065 - 178.43145161
    pas[pas < 0] = 0
    return pas


def convert_laser_aov(laser_aov: np.ndarray) -> np.ndarray:
    """
    Converts between the analog control voltage supplied to the laser
    and the power at sample
    :param laser_aov: The analog control voltage in V (0-10)
    :return: The power at sample
    """
    # if np.any(laser_aov < 0) or np.any(laser_aov > 10):
    #     raise ValueError("All laser control voltages must be between 0 and 10")
    laser_aov[laser_aov < 0] = 0
    laser_aov[laser_aov > 10] = 10
    current = laser_aov / 10 * 4000
    return convert_laser_current(current)


def predict_temperature(power_at_sample: np.ndarray, alpha=0.8787, beta=0.0356, dt=0.05):
    """
    Converts a laser power at sample trace to a temperature trace using a pre-determined heating model
    :param power_at_sample: Trace of power at sample values in mW
    :param alpha: The heating rate in K/J
    :param beta: The cooling rate in 1/s
    :param dt: The size of each timestep in power_at_sample in s
    :return: A delta-temperature trace
    """
    temps = np.zeros_like(power_at_sample)
    for t in range(1, power_at_sample.size):
        temps[t] = temps[t-1] + dt*(alpha*power_at_sample[t]/1000.0 - beta*temps[t-1])  # convert to W!!
    return temps


@njit
def interpolate_rows(x: np.ndarray, xp: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Performs linear interpolation on all rows of a matrix
    :param x: The desired time-base
    :param xp: The original time-base of each row in m
    :param m: The data matrix to interpolate row-wise
    :return: Interpolation of m with shape (m.shape[0], x.size)
    """
    out = np.zeros((m.shape[0], x.size))
    for i, r in enumerate(m):
        out[i, :] = np.interp(x, xp, r)
    return out


def bin_bout_starts(bout_data: np.ndarray, behav_times: np.ndarray, i_times: np.ndarray) -> np.ndarray:
    """
    Uses time-information of bout start and desired times to bin a behavior trace to Ca times
    :param bout_data: n_bouts x 8 matrix of swim bout information (column 0 = start frame)
    :param behav_times: for each behavioral frame the time relative to scan time
    :param i_times: The desired, even-spaced time-bins for which to record the number of bouts
    :return: i_times long vector with bout count in each frame
    """
    if np.std(np.diff(i_times)) >= 1e-5*np.mean(np.diff(i_times)):
        raise ValueError("i_times should be even-spaced")
    valid = np.logical_not(np.isnan(bout_data[:, 0]))
    bout_data = bout_data[valid, :]
    start_times = behav_times[bout_data[:, 0].astype(int)]
    start_times = start_times[start_times >= 0]  # remove bouts that occured before scanning started
    start_times = (start_times // np.diff(i_times)[0]).astype(int)  # these are now the actual bin indices into i_times
    out = np.zeros(i_times.size)
    ctr = Counter(start_times)
    for bin_ix in ctr.keys():
        if bin_ix >= out.size:
            # following indices would occur after scanning stopped
            break
        out[bin_ix] = ctr[bin_ix]
    return out


def bin_time_discrete(data_vector: np.ndarray, scan_times: np.ndarray, start_indices: Optional[np.ndarray],
                      end_indices: Optional[np.ndarray], i_times: np.ndarray) -> np.ndarray:
    """
    Uses time information of discrete data to create a behavior trace aligned to Ca times
    :param data_vector: the continuous metric
    :param scan_times: the scan frame times
    :param start_indices: the bout start indices at original frame rate
    :param end_indices: the bout end indices at original frame rate
    :param i_times: The desired, even-spaced time-bins for which to bin the input data
    :return: i_times long vector with binned data in each frame
    """
    if start_indices is None or start_indices.size == 0:
        return np.zeros(i_times.size)
    if data_vector is None or data_vector.size == 0:
        return np.zeros(i_times.size)
    assert data_vector.size == scan_times.size
    start_times = scan_times[start_indices]
    # remove events before scan start and thos at or after the end of the trace
    valid = np.logical_and(start_times >= 0, start_indices < scan_times.size-1)
    end_indices[end_indices > scan_times.size] = scan_times.size  # cut bouts at end of scanning
    if np.sum(valid) == 0:
        return np.zeros(i_times.size)
    start_indices = start_indices[valid]
    start_times = start_times[valid]
    end_indices = end_indices[valid]
    data_indeces = (start_times // np.diff(i_times)[0]).astype(int)  # these are now the actual bin indices into i_times
    out = np.zeros(i_times.size)

    for si, ei, di in zip(start_indices, end_indices, data_indeces):
        if di >= out.size:
            break
        out[di] = np.sum(data_vector[si:ei])
    return out


def prepare_motor_predictors(motor_data: Any, plane_id: int, i_times: np.ndarray,
                             bout_data: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
    """
    Takes motor model data and generates the appropriate motor predictors by binning tail metrics within bouts
    :param motor_data: Dataframe with all motor features
    :param plane_id: The number of the plane under consideration
    :param i_times: The interpolation times to align with imaging data
    :param bout_data: n_bouts x 8 matrix of swim bout information (column 0 = start frame, 1 = end frame)
    :return:
        [0]: n_predictors long list of i_times long vectors with the motor predictors
        [1]: n_predictors long list of the predictor names
    """
    predictor_names = []
    # df_names = ['rolling_vigor', 'delta_tail', 'sum_tail']
    df_names = ['rolling_vigor', 'sum_tail']
    # extract specifically the data in the current plane
    sub_frame = motor_data[motor_data['plane'] == plane_id]
    times = np.array(sub_frame['ts'])
    predictors = []
    for dfn in df_names:
        series = np.array(sub_frame[dfn])
        predictors.append(bin_time_discrete(series, times, bout_data[:, 0].astype(int), bout_data[:, 1].astype(int),
                                            i_times))
        predictor_names.append(dfn)
    return predictors, predictor_names


def analyze_experiment(file_path: str, file_name: str, use_dff: bool, overwrite: bool,
                       shuffle: bool, ridge_penalty: float) -> Union[int, str]:
    """
    Analyzes one experiment (fitting and evaluating ANN models)
    :param file_path: The path of the Experiment2P storage object
    :param file_name: The filename of the Experiment2P storage object
    :param use_dff: If true, anayze raw dff traces instead of calcium traces
    :param overwrite: If true truncate previous analysis and re-analyze
    :param shuffle: If true, roll calcium traces one trial forward
    :param ridge_penalty: If <=0, use plain linear regression. If >0 set alpha of ridge to this value
    :return: The number of found units or an information message if experiment was skipped
    """
    # create directory to store the analysis information for this experiment
    ana_dir_name = file_name[:file_name.find('.hdf5')]
    if shuffle:
        ana_dir_name += "_shuffle"
    ana_dir = path.join(file_path, ana_dir_name)
    behav_dframe_file = file_name[:file_name.find(".hdf5")] + ".bhvr"
    open_flag = 'w-'  # fail if hdf5 file already exists
    if not path.exists(ana_dir):
        os.mkdir(ana_dir)
    elif not overwrite:
        return f"Experiment {file_name} skipped. Analysis folder already exists."
    else:
        open_flag = 'w'  # overwrite hdf5 file if it already exists
    all_calcium_traces = []  # for each plane the calcium traces - note these are NOT the same length across planes!!!
    all_behav_predictors = []  # for each plane the behavior predictors - a list of per-plane lists!!
    predictor_names = None  # we assume that while predictors change across planes the predictors used do not
    all_temp_traces = []  # for each plane the one stimulus trace in delta-temperature
    with Experiment2P(path.join(file_path, file_name), False) as exp:
        # NOTE: Technically the interpolation is not required here since all experiments were
        # acquired at the same frame-rate. So instead we could just bring everything into the
        # actual experimental time-frame. But 5Hz has a nice correspondence with readable
        # time-intervals, so we still do it
        if not use_dff:
            all_ca = exp.all_c
        else:
            all_ca = exp.all_dff
        all_comp_brightness = exp.avg_component_brightness(False)
        all_vglut_brightness = exp.avg_component_brightness(True)
        for plane in range(exp.n_planes):
            # calcium data
            ca = all_ca[plane]
            if shuffle:
                # roll calcium data one trial forward (i.e. T1->T2, T2->T3, T3->T1)
                trial_len = ca.shape[1]//3
                ca = np.roll(ca, trial_len, 1)
            ca_frame_times = np.arange(ca.shape[1]) * exp.info_data["frame_duration"]
            interp_times = np.arange((ca_frame_times[-1] * 1000) // 200) * 0.2
            i_ca = interpolate_rows(interp_times, ca_frame_times, ca)
            all_calcium_traces.append(i_ca)
            # detailed tail behavior data
            behav_data_frame = pd.read_hdf(path.join(file_path, behav_dframe_file))
            bpreds, pn = prepare_motor_predictors(behav_data_frame, plane, interp_times, exp.bout_data[plane])
            # bout data
            b_starting = bin_bout_starts(exp.bout_data[plane], exp.tail_frame_times[plane], interp_times)
            bpreds = [b_starting] + bpreds
            pn = ["bout_start"] + pn
            if predictor_names is None:
                predictor_names = ["Temperature"] + pn  # This will be the overall ordering of predictors
            all_behav_predictors.append(bpreds)
            # stimulus data
            stim_aov = exp.laser_data[plane]
            stim_times = np.arange(stim_aov.size) / 20  # stimulus is stored at 20Hz
            stim_power = convert_laser_aov(stim_aov)
            stim_temp = predict_temperature(stim_power)
            i_stim_temp = np.interp(interp_times, stim_times, stim_temp)
            all_temp_traces.append(i_stim_temp)

    del bpreds  # THIS VARIABLE NAME IS WAY TOO SIMILAR TO b_preds used below so get rid of it...

    hist_steps = 5 * 10  # 4s history at 5Hz
    n_epochs = 100
    correlations_trained = []  # for each cell the training correlation
    correlations_test = []  # for each cell the test correlation
    lm_correlations_test = []  # for each cell test correlation on comparison linear model
    plane_ids = []  # for each cell the index of the plane it came from
    cell_indices = []  # for each cell it's within-plane index

    n_above_t = 0
    lm_n_above_t = 0

    with h5py.File(path.join(ana_dir, f"{ana_dir_name}_ANN_analysis.hdf5"), open_flag) as dfile:
        dfile.create_dataset(name="use_dff", data=use_dff)
        dfile.create_dataset(name="n_planes", data=len(all_calcium_traces))
        # loop over planes
        for p_id, (c_traces, b_preds, stim_trace, acb, vgl) in enumerate(zip(all_calcium_traces, all_behav_predictors,
                                                                             all_temp_traces, all_comp_brightness,
                                                                             all_vglut_brightness)):
            group = dfile.create_group(f"{p_id}")
            group.create_dataset(name="avg_comp_brightness", data=acb)
            group.create_dataset(name="anatomy_brightness", data=vgl)
            group.create_dataset(name="raw_stimulus", data=stim_trace+22)  # 22C = baseline temperature
            behavior = np.vstack(b_preds)
            group.create_dataset(name="raw_behavior", data=behavior)  # one ROW per predictor
            # to reduce the effect of mean-removal making vectors more dependent again (scalar subtractio is non-linear)
            # first subtract out the mean, then orthogonalize and then re-standardize
            behavior -= np.mean(behavior, axis=1, keepdims=True)
            # to remove dependencies, perform gram-schmidt orthogonalization, setting linear dependent vectors to 0
            behavior_gs = utilities.modified_gram_schmidt(behavior.T).T
            group.create_dataset(name="behavior_gs", data=behavior_gs, compression="gzip", compression_opts=5)
            c_traces = utilities.safe_standardize(c_traces, axis=1)
            c_traces = gaussian_filter1d(c_traces, 2.5, 1)
            b_preds = [utilities.safe_standardize(bp) for bp in behavior_gs]  # standardize each gs behavior predictor
            stim_trace = utilities.safe_standardize(stim_trace)
            # create our data object for training / testing
            train_frames = (c_traces.shape[1]*2)//3
            # predictor order: Stimulus first, followed by the behavior predictors
            data = utilities.Data(hist_steps, [stim_trace] + b_preds, c_traces, train_frames)
            # serialize whole data object into the plane group
            data.save_direct(group, False)  # if asked to overwrite above the file should have been truncated
            # obtain time-shifted predictor matrix for linear regression comparison in this plane
            lm_ts_predictors = utilities.create_time_shifted_predictor_matrix(np.vstack([stim_trace] + b_preds).T,
                                                                              hist_steps)
            group.create_dataset(name="lm_ts_predictors", data=lm_ts_predictors, compression="gzip", compression_opts=5)
            # write hdf5 file to disk in attempt to reclaim memory
            dfile.flush()
            # create our comparison model for this plane - re-use to avoid memory leak
            if ridge_penalty <= 0:
                lm = LinearRegression()
            else:
                lm = Ridge(alpha=ridge_penalty)

            for cell_ix in range(c_traces.shape[0]):
                if acb[cell_ix] < 0.1:
                    plane_ids.append(p_id)
                    cell_indices.append(cell_ix)
                    correlations_trained.append(np.nan)
                    correlations_test.append(np.nan)
                    lm_correlations_test.append(np.nan)
                    continue
                model_save_name = path.join(ana_dir, f"M_plane_{p_id}_cell_{cell_ix}_trained")
                plane_ids.append(p_id)
                cell_indices.append(cell_ix)
                tset = data.training_data(cell_ix, batch_size=256)
                m = model.get_standard_model(hist_steps)
                # the following is required to init variables at desired shape
                m(np.random.randn(1, hist_steps, len(data.regressors)).astype(np.float32))
                # train
                model.train_model(m, tset, n_epochs, c_traces.shape[1])
                m.save_weights(model_save_name)
                # evaluate
                p, r = data.predict_response(cell_ix, m)
                c_tr = np.corrcoef(p[:train_frames], r[:train_frames])[0, 1]
                correlations_trained.append(c_tr)
                c_ts = np.corrcoef(p[train_frames:], r[train_frames:])[0, 1]
                if c_ts >= 0.6:
                    n_above_t += 1
                correlations_test.append(c_ts)
                # compute test correlation for linear model - NOTE: To speed up, consider collating all outputs and
                # then fit as multi-target model, then predict and compute all r values
                c_for_lm = c_traces[cell_ix, :][hist_steps-1:]
                lm.fit(lm_ts_predictors[:train_frames, :], c_for_lm[:train_frames])
                y_hat = lm.predict(lm_ts_predictors[train_frames:, :]).ravel()
                lm_c_ts = np.corrcoef(y_hat, c_for_lm[train_frames:])[0, 1]
                if lm_c_ts >= 0.6:
                    lm_n_above_t += 1
                lm_correlations_test.append(lm_c_ts)
            print(f"Finished plane {p_id}. Until now, found {n_above_t} units with ANN, {lm_n_above_t} units with LM.")
        correlations_test = np.hstack(correlations_test)
        correlations_trained = np.hstack(correlations_trained)
        lm_correlations_test = np.hstack(lm_correlations_test)
        plane_ids = np.hstack(plane_ids)
        cell_indices = np.hstack(cell_indices)
        dfile.create_dataset(name="correlations_test", data=correlations_test)
        dfile.create_dataset(name="correlations_trained", data=correlations_trained)
        dfile.create_dataset(name="lm_correlations_test", data=lm_correlations_test)
        dfile.create_dataset(name="plane_ids", data=plane_ids)
        dfile.create_dataset(name="cell_indices", data=cell_indices)
        dfile.create_dataset(name="predictor_names", data=np.vstack([np.string_(s) for s in predictor_names]))
    return n_above_t


def main(data_dir: str, n_proc: int, use_dff: bool, overwrite: bool, shuffle: bool, ridge_penalty: float) -> None:
    """
    Runs the main loop of the script
    :param data_dir: The directory with the closed-loop data
    :param n_proc: The number of processes to run in parallel
    :param use_dff: If tue caiman's dff instead of calcium will be used as activity signal
    :param overwrite: Whether to reanalyze or skip previously analyzed experiments
    :param shuffle: If true, roll calcium traces one trial forward
    :param ridge_penalty: If <=0, use plain linear regression. If >0 set alpha of ridge to this value
    :return: None
    """
    pool = Pool(n_proc, maxtasksperchild=1)  # limit tasks that a single process runs to 1 to frequently clean up memory
    file_list = [f for f in os.listdir(data_dir) if ".hdf5" in f]
    a_results = []
    for exp_name in file_list:
        a_results.append(pool.apply_async(analyze_experiment, [data_dir, exp_name, use_dff, overwrite, shuffle,
                                                               ridge_penalty]))
    for i, a in enumerate(a_results):
        units_found = a.get()
        print(f"Analysis on experiment {i+1} out of {len(file_list)} experiments completed.")
        print(f"Found {units_found} units with above-threshold test correlations.")
        print()


if __name__ == "__main__":

    # if the unix-default "fork" is used we cannot properly set maxtasksperchild in pool creation above
    # therefore force process creation as 'spawn' (windows and osx default)
    mp.set_start_method('spawn')

    # the following will prevent tensorflow from using the GPU - as the used models have very low complexity
    # they will generally be fit faster on the CPU - furthermore parallelization currently used
    # will not work if tensorflow is run on the GPU!!
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    a_parser = argparse.ArgumentParser(prog="rwave_data_fit",
                                       description="Uses predictor and calcium imaging data in experiments"
                                                   " analyzed by imaging_pipeline to fit ANN models.")
    a_parser.add_argument("-f", "--folder", help="Path to folder with experiment hdf5 files", type=str, default="",
                          action=CheckArgs)
    a_parser.add_argument("-dff", "--use_dff", help="Whether to analyze raw dff (set) or deconvolved calcium traces.",
                          action='store_true')
    a_parser.add_argument("-np", "--processes", help="The number of processes across which to parallelize computation.",
                          type=int, default=1, action=CheckArgs)
    a_parser.add_argument("-ovr", "--overwrite", help="If set analyzed experiments won't be skipped but re-analyzed",
                          action='store_true')
    a_parser.add_argument("-shf", "--shuffle", help="If set calcium data will be rolled one trial forward before"
                                                    " fitting", action='store_true')
    a_parser.add_argument("-rp", "--ridge_penalty", help="Value of ridge alpha for linear comparison model. <=0 reverts"
                                                         " to ordinary linear regression.", type=float, default=1e-4)

    args = a_parser.parse_args()

    data_folder = args.folder
    dff = args.use_dff
    n_processes = args.processes
    ovr = args.overwrite
    shf = args.shuffle
    ridgp = args.ridge_penalty

    we_process = "DFF" if dff else "Calcium"
    print(f"Analyzing experiments in folder {data_folder}. Using {n_processes} processes. Processing "
          f"{we_process} signals")
    if ridgp <= 0:
        print("Comparison model uses ordinary linear regression.")
    else:
        print(f"Comparison model uses Ridge regression with alpha={ridgp}")
    main(data_folder, n_processes, dff, ovr, shf, ridgp)
