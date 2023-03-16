"""
Script to plot paper figure panels from main analysis file generated via rwave_build_main.py
"""

import argparse
import os
from os import path
from typing import Any, List
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl
from matplotlib_venn import venn2
import seaborn as sns
from statsmodels.tsa.stattools import acf
from pandas import DataFrame
import rwave_build_main as rbm
import upsetplot as ups
import utilities
import model
from sklearn.cluster import SpectralClustering
import AnatomyClustering as acl
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
        if self.dest == 'file':
            if not path.exists(values):
                raise argparse.ArgumentError(self, f"Specified file {values} does not exist")
            if not path.isfile(values):
                raise argparse.ArgumentError(self, f"The destination {values} is not a file")
            setattr(namespace, self.dest, values)
        else:
            raise Exception("Parser was asked to check unknown argument")


def plot_region_enrichment(c_of_interest: np.ndarray, population: np.ndarray, masks: h5py.Group,
                           outline_top: np.ndarray, outline_side: np.ndarray, name: str,
                           region_list: List[str], r_memship: np.ndarray) -> None:
    """
    For a given cluster plot its prevalence in brain regions and maps of the top-five regions
    :param c_of_interest: n_fit long boolean vector marking membership in the class of interest
    :param population: n_fit long boolean vector marking membership in all classes that are considered
    :param masks: hdf5 object that contains mask outlines
    :param outline_top: Overall brain or region outline to draw in the background, top projection
    :param outline_side: Overall brain or region outline to draw in the background, side projection
    :param name: Name of the group to add to plot saves
    :param region_list: The list of all regions to consider
    :param r_memship: n_fit x len(region_list) matrix marking membership in regions for each neuron
    """
    global ext
    global plot_dir
    global rgnn_abbreviations
    assert population.size == c_of_interest.size
    assert population.size == r_memship.shape[0]
    assert len(region_list) == r_memship.shape[1]

    # get the total number of above-threshold units in each region for the population
    pop_in_regions = np.sum(r_memship[population], 0)
    r_valid = pop_in_regions >= 50  # filter out regions in which we identified less than 50 neurons
    pop_in_regions = pop_in_regions[r_valid]
    r_labels = [rn for k, rn in enumerate(region_list) if r_valid[k]]
    sel_in_regions = np.sum(r_memship[c_of_interest], 0)
    sel_in_regions = sel_in_regions[r_valid]
    sel_frac = sel_in_regions / pop_in_regions
    # create rough anterior->posterior sort index for region display
    sort_value = np.zeros(sel_frac.size)
    for index, rl in enumerate(r_labels):
        if "Telencephalon" in rl:
            sort_value[index] = 1
        elif "Diencephalon" in rl:
            sort_value[index] = 2
        elif "Mesencephalon" in rl:
            sort_value[index] = 3
        elif "Rhombencephalon" in rl:
            sort_value[index] = 4
    sel_sort = np.argsort(sort_value, kind='stable')[::-1]

    # computed expected quantity and for each region based on the number
    # of neurons in there (pop_in_region) a confidence interval
    expectation = c_of_interest.sum() / population.sum()
    exp_lower_95 = np.zeros(sel_sort.size)
    exp_upper_95 = np.zeros_like(exp_lower_95)
    for i, psize in enumerate(pop_in_regions):
        ran_mat = np.random.rand(psize, 5000) < expectation  # roll dice with p(heads)=expectation
        ran_fracs = np.sum(ran_mat.astype(float), 0) / psize
        exp_lower_95[i] = np.percentile(ran_fracs, 2.5)
        exp_upper_95[i] = np.percentile(ran_fracs, 97.5)

    sorted_labels = [r_labels[k] for k in sel_sort]
    sorted_abb = [rgnn_abbreviations[sl] for sl in sorted_labels]
    sorted_frac = sel_frac[sel_sort]
    sorted_upper = exp_upper_95[sel_sort]

    figure = pl.figure(figsize=[13, 7.5])
    sns.barplot(y=sorted_abb, x=sorted_frac, color='grey')
    sns.pointplot(y=sorted_abb, x=exp_lower_95[sel_sort], color='C0', alpha=0.5, markers="+", join=False)
    sns.pointplot(y=sorted_abb, x=np.ones(sel_sort.size) * expectation, color='C0', markers="None")
    sns.pointplot(y=sorted_abb, x=sorted_upper, color='C0', alpha=0.5, markers="+", join=False)
    sns.despine()
    figure.tight_layout()
    figure.savefig(path.join(plot_dir, f"ZBrainRegions_Fraction_{name}Neurons.{ext}"), dpi=300)


def plot_autocorrelations(in_mat: np.ndarray, lags: int, name: str) -> None:
    """
    Plot autocorrelations for multiple traces
    :param in_mat: n_traces x m_timepoints matrix of timeseries
    :param lags: The lags (in frames) to consider
    :param name: Name to add to plot-saves
    """
    t_lags = np.arange(lags + 1) / 5
    all_acf = []
    for trace in in_mat:
        if np.allclose(trace, np.mean(trace)):
            continue
        all_acf.append(acf(trace, nlags=lags))
    all_acf = np.vstack(all_acf)
    avg_acf = np.mean(all_acf, 0)
    figure = pl.figure()
    for ac in all_acf:
        pl.plot(t_lags, ac, 'C1', alpha=0.1, lw=0.5)
    pl.plot(t_lags, avg_acf, 'k')
    pl.xlabel("Lag [s]")
    pl.ylabel("Autocorrelation")
    sns.despine(fig)
    figure.savefig(path.join(plot_dir, f"Autocorrelation_{name}.{ext}"), dpi=300)


def boot_complexity(all_scores: np.ndarray, subset_size: int, ci: float, n_boot: int):
    """
    Uses bootstrapping to obtain a confidence interval of average complexity
    :param all_scores: Complexity scores of all considered units
    :param subset_size: The size of the particular subset
    :param ci: The intended confidence interval in percent
    :param n_boot: The number of bootstrap samples to generate
    :return:
        [0]: Lower bound according to ci
        [1]: Upper bound according to ci
        [2]: Averaage complexity of bootstrap sample
    """
    if ci <= 0 or ci >= 100:
        raise ValueError(f"Confidence interval of {ci} is not valid. Has to be 0 < ci < 100")
    samples = np.full(n_boot, np.nan)
    for bs in range(n_boot):
        subset = np.random.choice(all_scores, subset_size, True)
        samples[bs] = np.mean(subset)
    return np.percentile(samples, (100 - ci) / 2), np.percentile(samples, (100 - ci) / 2 + ci), np.mean(samples)


def get_neuron_model_weights(neuron_id: int, neuronlist: DataFrame, experimentlist: DataFrame,
                             mfilepath: str) -> List[np.ndarray]:
    """
    For a given neuron returns the weights of the fit model of that neuron
    :param neuron_id: The index of the neuron in question
    :param neuronlist: The neuron-list dataframe
    :param experimentlist: The experimet-list dataframe
    :param mfilepath: Path to the directory with the model files
    :return: Weightlist of the model
    """
    all_neuron_ids = np.hstack(neuronlist["neuron_id"])  # the ids of all neurons
    all_plane_ids = np.hstack(neuronlist["plane"])  # all plane ids for each neuron across all experiments
    all_experiment_ids = np.hstack(neuronlist["experiment_id"])  # all experiment ids for each neuron
    this_neuron_plane = all_plane_ids[all_neuron_ids == neuron_id][0]  # imaging plane of neuron in question
    ex_id = all_experiment_ids[all_neuron_ids == neuron_id]  # the experiment id of this neuron
    # Count how many neurons are in total in the plane and experiment of the neuron in question
    neuron_plane_count = np.sum(np.logical_and(all_plane_ids == this_neuron_plane, all_experiment_ids == ex_id))
    # Get the indices of all the neurons within the same plane of the same experiment
    neurons_in_plane = all_neuron_ids[np.logical_and(all_plane_ids == this_neuron_plane, all_experiment_ids == ex_id)]
    # Find within the given plane and experiment what the index is of the neuron under consideration
    this_neuron_in_plane_id = np.arange(neuron_plane_count)[neurons_in_plane == neuron_id][0]  # original cell index
    all_experiment_ids = np.hstack(experimentlist["experiment_id"])
    experiment_name = np.hstack(experimentlist["experiment_name"])[all_experiment_ids == ex_id][0]
    file_name = experiment_name.decode("UTF-8") + "_fit_models.hdf5"
    with h5py.File(path.join(mfilepath, file_name), 'r') as mfile:
        model_save_group = f"M_plane_{this_neuron_plane}_cell_{this_neuron_in_plane_id}_trained"
        return utilities.modelweights_from_hdf5(mfile[model_save_group])


def get_neuron_model(neuron_id: int, neuronlist: DataFrame, experimentlist: DataFrame,
                     mfilepath: str) -> model.ActivityPredictor:
    """
    For a given neuron returns the model that was fit to that neuron
    :param neuron_id: The index of the neuron in question
    :param neuronlist: The neuron-list dataframe
    :param experimentlist: The experimet-list dataframe
    :param mfilepath: Path to the directory with the model files
    :return: The CNN model
    """
    m_weights = get_neuron_model_weights(neuron_id, neuronlist, experimentlist, mfilepath)
    m = model.get_standard_model(m_weights[0].shape[0])
    init_in = np.random.randn(1, m_weights[0].shape[0], m_weights[0].shape[1]).astype(np.float32)
    m(init_in)
    m.set_weights(m_weights)
    m(init_in)
    return m


def get_drive_outputs(netw: model.ActivityPredictor, netw_j: np.ndarray, ix_i1: int, ix_i2: int, drive_min: float,
                      drive_max: float, n_drives: int) -> np.ndarray:
    """
    Uses a network's data-mean jacobian to create 2D variatins in input drive and computes the corresponding model
    outputs
    :param netw: The model
    :param netw_j: The jacobian of the model at the original data mean (receptive field)
    :param ix_i1: The index of the first input for which drive should be varied
    :param ix_i2: The index of the second input for which drive should be varied
    :param drive_min: The minimal drive to use
    :param drive_max: The maximal drive to use
    :param n_drives: The number of discrete drive values along each axis
    :return: n_drives x n_drives matrix of network outputs
    """
    hist = netw.input_length
    drives = np.linspace(drive_min, drive_max, n_drives)
    n_inputs = netw_j.size // hist
    i_base = np.full((1, hist, n_inputs), np.nan)
    for i in range(n_inputs):
        i_base[0, :, i] = netw_j.ravel()[i*hist:(i+1)*hist]
    i_base[:, :, ix_i1] = i_base[:, :, ix_i1]/np.linalg.norm(i_base[:, :, ix_i1])
    i_base[:, :, ix_i2] = i_base[:, :, ix_i2]/np.linalg.norm(i_base[:, :, ix_i2])
    m_output = np.full((n_drives, n_drives), np.nan)
    for i, drive1 in enumerate(drives):
        for j, drive2 in enumerate(drives):
            i_current = i_base.copy()
            i_current[:, :, ix_i1] = i_current[:, :, ix_i1] * drive1
            i_current[:, :, ix_i2] = i_current[:, :, ix_i2] * drive2
            m_output[i, j] = netw(i_current)
    return m_output


def plot_drive_response(drive_mat):
    """
    Plots the drive response as a heatmap as well as three cross-sections
    :param drive_mat: The sensory vs. motor drive response of a neuron
    :return: The figure object for saving
    """
    dfig, (ax1, ax2) = pl.subplots(ncols=2)
    sns.heatmap(data=drive_mat, ax=ax1, cmap='inferno')
    ax2.plot(drive_mat[:, 24], label="Motor drive = -5")
    ax2.plot(drive_mat[:, 49], label="Motor drive = 0")
    ax2.plot(drive_mat[:, 74], label="Motor drive = 5")
    ax2.legend()
    return dfig


if __name__ == "__main__":
    mpl.rcParams['pdf.fonttype'] = 42

    a_parser = argparse.ArgumentParser(prog="rwave_plot_panels",
                                       description="Will plot all RandomWave experiment figure panels")
    a_parser.add_argument("-f", "--file", help="Path to the main analysis hdf5 file", type=str, default="",
                          action=CheckArgs)
    a_parser.add_argument("-png", "--save_png", help="If set figures will be saved as png instead of pdf",
                          action='store_true')

    cl_args = a_parser.parse_args()

    data_file = cl_args.file
    as_png = cl_args.save_png

    ext = "png" if as_png else "pdf"

    data_folder, data_filename = path.split(data_file)
    base_name = path.splitext(data_filename)[0]

    model_file_folder = path.join(data_folder, "fit_models")

    plot_dir = path.join(data_folder, f"{base_name}_figure_panels")
    if not path.exists(plot_dir):
        os.makedirs(plot_dir)

    with h5py.File(data_file, 'r') as dfile:
        fit_threshold = dfile["ANN test correlation threshold"][()]  # should be R^2=0.5, 50% variance explained
        if not np.isclose(fit_threshold, np.sqrt(0.5)):
            print("Data in file was analyzed with different fit threshold than intended for publication")
        if not np.isclose(fit_threshold, dfile["Linear model threshold"][()]):
            print("Linear and ANN model thresholds used in analysis file are different from each other")
        lscore_th = dfile["Linear model score threshold"][()]  # should be >=80% of explained variance to be linear
        if not np.isclose(lscore_th, 0.8):
            print("Data in file was analyzed with different linearity threshold than intended for publication")
        me_threshold = dfile["Mean expansion R2 threshold"][()]  # me-scores are already in the form of R^2 values
        if not np.isclose(me_threshold, 0.5):
            print("Data in file was analyzed with different mean expansion threshold than intended for publication")

        barcode_labels = dfile["Barcode labels"]
        barcode_labels = [bl[0].decode('utf-8') for bl in barcode_labels]

        #######################################
        # Raw data analyses
        #######################################

        df_raw = rbm.get_raw_data(dfile)
        all_test_corrs = np.hstack(df_raw["ann_corr"])
        all_test_corrs_sh = np.hstack(df_raw["ann_sh_corr"])
        all_lm_test_corrs = np.hstack(df_raw["lm_corr"])
        above_thresh = np.hstack(df_raw["is_ann_fit"])
        above_threshl_lm = np.hstack(df_raw["is_lm_fit"])

        # plot venn diagram comparing ann and linear model identification
        id_ann = np.sum(above_thresh)
        id_lm = np.sum(above_threshl_lm)
        overlap = np.sum(np.logical_and(above_thresh, above_threshl_lm))
        total = np.sum(np.logical_or(above_thresh, above_threshl_lm))
        fig = pl.figure()
        venn2(subsets=(np.round((id_ann - overlap) / total, 2),
                       np.round((id_lm - overlap) / total, 2),
                       np.round(overlap / total, 2)), set_labels=("ANN", "LM"), set_colors=("C3", "C0"))
        fig.savefig(path.join(plot_dir, f"ANN_LM_Venn.{ext}"), dpi=300)

        # across thresholds plot comparison of true and shuffle ID of ANN
        thresh_to_test = np.linspace(0, 1, 101)
        abt_real_ann = np.zeros_like(thresh_to_test)
        abt_sh_ann = np.zeros_like(thresh_to_test)
        abt_real_lm = np.zeros_like(thresh_to_test)
        for i, ttt in enumerate(thresh_to_test):
            abt_real_ann[i] = np.sum(all_test_corrs >= ttt)
            abt_sh_ann[i] = np.sum(all_test_corrs_sh >= ttt)
            abt_real_lm[i] = np.sum(all_lm_test_corrs >= ttt)
        maxval = max([np.max(abt_real_ann), np.max(abt_sh_ann)])
        enrich_at_th = np.round(np.sum(all_test_corrs >= fit_threshold) / np.sum(all_test_corrs_sh >= fit_threshold))
        fig = pl.figure()
        pl.plot(thresh_to_test, abt_real_ann, label="Real data")
        pl.plot(thresh_to_test, abt_sh_ann, label="Rotated data")
        pl.plot([fit_threshold, fit_threshold], [0, maxval], '--', label=f"{enrich_at_th} x enrichment")
        pl.xlabel("Test correlation threshold")
        pl.ylabel("N above threshold")
        pl.ylim(0, maxval)
        sns.despine(fig)
        pl.legend()
        fig.tight_layout()
        fig.savefig(path.join(plot_dir, f"ANN_Shuffle_Threshold_Comparison.{ext}"), dpi=300)

        # across thresholds plot comparison of ID by ANN and LM
        fig = pl.figure()
        pl.plot(thresh_to_test, abt_real_ann, label="ANN")
        pl.plot(thresh_to_test, abt_real_lm, label="Linear Model")
        pl.plot([fit_threshold, fit_threshold], [0, maxval], '--')
        pl.xlabel("Test correlation threshold")
        pl.ylabel("N above threshold")
        pl.ylim(0, maxval)
        sns.despine(fig)
        pl.legend()
        fig.tight_layout()
        fig.savefig(path.join(plot_dir, f"ANN_LM_Threshold_Comparison.{ext}"), dpi=300)

        # plot nonlinearity model probabilities
        l_approx_score = np.hstack(df_raw["linear_model_score"])[above_thresh]
        fig = pl.figure()
        sns.kdeplot(l_approx_score)
        pl.plot([lscore_th, lscore_th], [0, 1], 'k--')
        pl.xlabel("Linear approximation [R2]")
        pl.ylabel("Density")
        sns.despine(fig)
        fig.savefig(path.join(plot_dir, f"Linear_approximation_score_Distribution.{ext}"), dpi=300)

        # plot relationship of nonlinearity probability and me-score
        me_score = np.hstack(df_raw["me_score"])[above_thresh]
        fig = pl.figure()
        pl.scatter(l_approx_score, me_score, s=2, alpha=0.25, c='C3')
        pl.plot([0, 1], [me_threshold, me_threshold], 'k--')
        pl.plot([lscore_th, lscore_th], [0, 1], 'k--')
        pl.text(0.25, 0.75, np.sum(np.logical_and(l_approx_score < lscore_th, me_score >= me_threshold)))
        pl.text(0.25, 0.25, np.sum(np.logical_and(l_approx_score < lscore_th, me_score < me_threshold)))
        pl.text(0.75, 0.75, np.sum(np.logical_and(l_approx_score >= lscore_th, me_score >= me_threshold)))
        pl.text(0.75, 0.25, np.sum(np.logical_and(l_approx_score >= lscore_th, me_score < me_threshold)))
        pl.xlim(-0.01, 1.01)
        pl.ylim(-0.01, 1.01)
        pl.xlabel("Linear approximation $R^2$")
        pl.ylabel("2nd order model $R^2$")
        sns.despine()
        fig.savefig(path.join(plot_dir, f"mescore_vs_linscore.{ext}"), dpi=300)

        #######################################
        # Paradigm analysis
        #######################################

        experiment_names = np.vstack(rbm.get_experiment_list(dfile)["experiment_name"])
        experiment_names = [en[0].decode('utf-8') for en in experiment_names]

        # plot autocorrelations across stimuli and behavior
        frame_lags = 5 * 30  # up to 30 seconds considered
        df_stimuli = rbm.get_stimuli(dfile)
        all_stimuli = np.vstack(df_stimuli["temperature"])
        df_behavior = rbm.get_behaviors(dfile)
        all_starts = np.vstack(df_behavior["bout_start"])
        all_rvig = np.vstack(df_behavior["rolling_vigor"])
        all_sumtail = np.vstack(df_behavior["sum_tail"])
        plot_autocorrelations(all_stimuli, frame_lags, "Temperature")
        plot_autocorrelations(all_starts, frame_lags, "BoutStarts")
        plot_autocorrelations(all_rvig, frame_lags, "RollingVigor")
        plot_autocorrelations(all_sumtail, frame_lags, "SumTail")

        # plot example traces - stimulus and behavior
        ix_chosen = 641
        fig = pl.figure()
        s = all_stimuli[ix_chosen]
        stim_time = np.arange(s.size) / 5
        pl.plot(stim_time, s)
        pl.plot([2 * (60 + 30 + 75), 2 * (60 + 30 + 75)], [22, 30], 'k--')
        pl.xticks([0, 125, 250, 375, 500])
        pl.ylim(20, 30)
        pl.yticks([20, 22.5, 25, 27.5, 30])
        pl.xlabel("Time [s]")
        fig.savefig(path.join(plot_dir, f"ExampleStimulus.{ext}"), dpi=300)

        # plot behavior examples for same experiments as stimulus examples
        fig, axes = pl.subplots(nrows=3)
        b = all_starts[ix_chosen]
        b_time = np.arange(b.size) / 5
        axes[0].vlines(b_time[b > 0], 0.25, 0.75, colors='C0')
        axes[0].plot([2 * (60 + 30 + 75), 2 * (60 + 30 + 75)], [0, 1], 'k--')
        axes[0].set_xticks([0, 125, 250, 375, 500])
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Swim")
        rv = all_rvig[ix_chosen]
        axes[1].plot(b_time, rv, 'C0')
        axes[1].plot([2 * (60 + 30 + 75), 2 * (60 + 30 + 75)], [rv.min(), rv.max()], 'k--')
        axes[1].set_xticks([0, 125, 250, 375, 500])
        axes[1].set_ylabel("Vigor")
        st = all_sumtail[ix_chosen]
        axes[2].plot(b_time, st, 'C0')
        axes[2].plot([2 * (60 + 30 + 75), 2 * (60 + 30 + 75)], [st.min(), st.max()], 'k--')
        axes[2].set_xticks([0, 125, 250, 375, 500])
        axes[2].set_ylabel("Directionality")
        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        fig.savefig(path.join(plot_dir, f"ExampleBehavior.{ext}"), dpi=300)

        # plot activity heatmap sorted by ANN fit score for one plane
        if "activity" in dfile:
            act_index = ix_chosen
            act_plane = df_behavior["plane"][act_index]
            act_exp = df_behavior["experiment_id"][act_index]
            print(f"Example plane is plane {act_plane} from {experiment_names[act_exp]}")
            df_activity = rbm.get_cadata(dfile)
            act_valid = np.logical_and(np.hstack(df_activity["experiment_id"]) == act_exp,
                                       np.hstack(df_activity["plane"]) == act_plane)
            act_sorting = np.argsort(all_test_corrs[np.isfinite(all_test_corrs)][act_valid])[::-1]
            fig = pl.figure()
            sns.heatmap(np.vstack(df_activity["ca_data"])[act_valid][act_sorting], vmax=8, cmap="bone", rasterized=True)
            pl.xlabel("Frame")
            pl.ylabel("Neuron")
            fig.savefig(path.join(plot_dir, f"ExampleActivity_SortedByDecreasingFit.{ext}"), dpi=600)
            fig = pl.figure()
            pl.plot(all_test_corrs[np.isfinite(all_test_corrs)][act_valid][act_sorting], 'k')
            pl.plot([0, np.sum(act_valid)], [0, 0], 'C0--')
            pl.plot([0, np.sum(act_valid)], [fit_threshold, fit_threshold], 'C3--')
            pl.xticks([0, 250, 500, 750, 1000])
            pl.yticks([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
            pl.ylim(-0.75, 1.0)
            pl.xlabel("Neuron number")
            pl.ylabel("ANN fit test correlation")
            sns.despine()
            fig.savefig(path.join(plot_dir, f"TestCorrs_ForExampleActivity.{ext}"), dpi=300)

        # plot distribution of correlations across the three trials for stimulus and behavior metrics
        c_inter_stim = []
        c_inter_starts = []
        c_inter_rvig = []
        c_inter_sumtail = []
        t_len = all_stimuli.shape[1] // 3  # note that this is not exact for all planes since we prune to shortest
        trials = [slice(0, t_len), slice(t_len, 2 * t_len), slice(2 * t_len, 3 * t_len)]
        for i in range(all_stimuli.shape[0]):
            this_stim = all_stimuli[i]
            this_starts = all_starts[i]
            this_rvig = all_rvig[i]
            this_sumtail = all_sumtail[i]
            for j, t1 in enumerate(trials):
                for t2 in trials[j + 1:]:
                    c_inter_stim.append(np.corrcoef(this_stim[t1], this_stim[t2])[0, 1])
                    c_inter_starts.append(np.corrcoef(this_starts[t1], this_starts[t2])[0, 1])
                    c_inter_rvig.append(np.corrcoef(this_rvig[t1], this_rvig[t2])[0, 1])
                    c_inter_sumtail.append(np.corrcoef(this_sumtail[t1], this_sumtail[t2])[0, 1])
        c_inter_stim = np.hstack(c_inter_stim)[:, None]
        c_inter_starts = np.hstack(c_inter_starts)[:, None]
        c_inter_rvig = np.hstack(c_inter_rvig)[:, None]
        c_inter_sumtail = np.hstack(c_inter_sumtail)[:, None]
        df_trial_correlations = DataFrame(np.hstack((c_inter_stim, c_inter_starts, c_inter_rvig, c_inter_sumtail)),
                                          columns=["Temperature", "Swim", "Vigor", "Directionality"])
        fig = pl.figure()
        sns.boxenplot(data=df_trial_correlations)
        pl.ylabel("Inter-trial correlations")
        sns.despine()
        fig.savefig(path.join(plot_dir, f"InterTrialCorrelations.{ext}"), dpi=300)

        #######################################
        # Analyses across ann fit units
        #######################################

        df_fit = rbm.get_fit_neurons(dfile)

        # plot barcode clustering upset plot
        barcodes = np.vstack(df_fit["barcode"])
        df_test_metric = DataFrame(barcodes[:, :6], columns=barcode_labels[:6])
        aggregate = ups.from_indicators(df_test_metric)
        b_levels = []
        fig = pl.figure()
        up_set = ups.UpSet(aggregate, subset_size='count', min_subset_size=10, facecolor="grey", sort_by='cardinality',
                           sort_categories_by=None)
        up_set.style_subsets(present=["behavior", "Temperature"], facecolor='C0', label="Mixed selectivity")
        b_levels += ["behavior"]
        up_set.style_subsets(present=["bout_start", "Temperature"], facecolor='C0')
        b_levels += ["bout_start"]
        up_set.style_subsets(present=["sum_tail", "Temperature"], facecolor='C0')
        b_levels += ["sum_tail"]
        up_set.style_subsets(present=["rolling_vigor", "Temperature"], facecolor='C0')
        b_levels += ["rolling_vigor"]
        up_set.style_subsets(present="Temperature", absent=b_levels,
                             facecolor="C1", label="Temperature")
        if "behavior" in b_levels:
            up_set.style_subsets(present="behavior", absent="Temperature", facecolor="w", edgecolor="C2",
                                 label="Behavior")
        if "bout_start" in b_levels:
            up_set.style_subsets(present="bout_start", absent="Temperature", facecolor="w", edgecolor="C2")
        if "sum_tail" in b_levels:
            up_set.style_subsets(present="sum_tail", absent="Temperature", facecolor="w", edgecolor="C2")
        if "rolling_vigor" in b_levels:
            up_set.style_subsets(present="rolling_vigor", absent="Temperature", facecolor="w", edgecolor="C2")
        axes_dict = up_set.plot(fig)
        axes_dict['intersections'].set_yscale('log')
        fig.savefig(path.join(plot_dir, f"BarcodeUpsetPlot.{ext}"), dpi=300)

        # for fish-contribution analysis convert barcode into unique cluster numbers
        bc_cluster_number = np.zeros(barcodes.shape[0])
        for i in range(6):
            bc_cluster_number += barcodes[:, i] * (2**i)
        # as in plot limit clusters by minimum size of 10, put all else into a "-1" cluster
        cnt = Counter(bc_cluster_number)
        # find the experiment id that belongs to each clustered neuron
        df_neuron_list = rbm.get_neuron_list(dfile)
        exp_fit = np.hstack(df_neuron_list["experiment_id"])[np.hstack(df_neuron_list["is_ann_fit"])]
        # neuron_fit_id = np.hstack(df_fit["neuron_id"])
        # print how many experiments contributed to each cluster sorted by cluster size (same order as in upset plot)
        # for all clusters with at least 10 neurons (same cutoff as in plot)
        unique_clusters = np.array([k for k in cnt])
        cluster_sizes = np.array([cnt[k] for k in cnt])
        size_sort = np.argsort(cluster_sizes)[::-1]
        cluster_sizes = cluster_sizes[size_sort]
        unique_clusters = unique_clusters[size_sort]
        for i, cnum in enumerate(unique_clusters):
            if cluster_sizes[i] < 10:
                break
            clus_exps = exp_fit[bc_cluster_number == cnum]
            print(f"Cluster {i} has {np.unique(clus_exps).size} contributing fish")

        nl_id = barcode_labels.index("Nonlin")
        t_id = barcode_labels.index("Temperature")
        ang_id = barcode_labels.index("sum_tail")
        disp_id = barcode_labels.index("rolling_vigor")
        start_id = barcode_labels.index("bout_start")
        # behavior is from the "behavior" catch all until the first interaction term
        any_b_id = slice(barcode_labels.index("behavior"), [loc for loc, s in enumerate(barcode_labels) if "I" in s][0])
        # interactors are at the end
        any_interact = slice([loc for loc, s in enumerate(barcode_labels) if "I" in s][0], barcodes.shape[1])

        non_linear = barcodes[:, nl_id] == 1
        complex_2 = np.hstack(df_fit["complexity"]) == 2
        linear = np.logical_not(non_linear)
        sens_contrib = barcodes[:, t_id] == 1
        mot_contrib = np.sum(barcodes[:, any_b_id] == 1, 1) >= 1
        int_contrib = np.sum(barcodes[:, any_interact], 1) >= 1

        non_sens_contrib = np.logical_or(mot_contrib, int_contrib)
        non_mot_contrib = np.logical_or(sens_contrib, int_contrib)

        sensory_lin_cluster = np.logical_and(np.logical_and(linear, sens_contrib), np.logical_not(non_sens_contrib))
        sensory_nl_cluster = np.logical_and(np.logical_and(non_linear, sens_contrib), np.logical_not(non_sens_contrib))
        sensory_cluster = np.logical_or(sensory_nl_cluster, sensory_lin_cluster)  # pure sensory

        motor_lin_cluster = np.logical_and(np.logical_and(linear, mot_contrib), np.logical_not(non_mot_contrib))
        motor_nl_cluster = np.logical_and(np.logical_and(non_linear, mot_contrib), np.logical_not(non_mot_contrib))
        motor_cluster = np.logical_or(motor_lin_cluster, motor_nl_cluster)  # pure motor

        interact_lin_cluster = np.logical_or(np.logical_and(linear, np.logical_and(sens_contrib, mot_contrib)),
                                             int_contrib)
        interact_nl_cluster = np.logical_or(np.logical_and(non_linear, np.logical_and(sens_contrib, mot_contrib)),
                                            int_contrib)
        interact_cluster = np.logical_or(interact_lin_cluster, interact_nl_cluster)

        df_experiment_list = rbm.get_experiment_list(dfile)

        all_jacs = np.vstack(df_fit['jacobian'])
        all_fit_neuron_ids = np.hstack(df_fit["neuron_id"])

        # Plot anatomical clustering of our types of interest
        all_centroids_um = np.vstack(df_neuron_list["zbrain_coords"])
        all_centroids_um = all_centroids_um[above_thresh]
        # By response type
        lcoors_sensory = acl.Cluster_Neurons(all_centroids_um[sensory_cluster], [5])[-1]
        lcoors_motor = acl.Cluster_Neurons(all_centroids_um[motor_cluster], [5])[-1]
        lcoors_interact = acl.Cluster_Neurons(all_centroids_um[interact_cluster], [5])[-1]
        # Neurons by complexity
        complexity = np.hstack(df_fit["complexity"])
        c0 = complexity == 0  # linear
        c1 = complexity == 1  # nonlinear and 2nd order model can fit response
        c2 = complexity == 2  # nonlinear and 2nd order model cannot fit response
        lcoors_comp0 = acl.Cluster_Neurons(all_centroids_um[c0], [5])[-1]
        lcoors_comp1 = acl.Cluster_Neurons(all_centroids_um[c1], [5])[-1]
        lcoors_comp2 = acl.Cluster_Neurons(all_centroids_um[c2], [5])[-1]
        # Motor types
        # create motor-type clusters and make them exclusive - otherwise in the region enrichments
        # below some regions can be enriched for all types, since the expectations underestimate
        ang_cluster = barcodes[:, ang_id] == 1
        disp_cluster = barcodes[:, disp_id] == 1
        start_cluster = barcodes[:, start_id] == 1
        start_cluster[np.logical_or(ang_cluster, disp_cluster)] = False
        ang_cluster[disp_cluster] = False
        lcoors_start = acl.Cluster_Neurons(all_centroids_um[start_cluster], [5])[-1]
        lcoors_disp = acl.Cluster_Neurons(all_centroids_um[disp_cluster], [5])[-1]
        lcoors_ang = acl.Cluster_Neurons(all_centroids_um[ang_cluster], [5])[-1]

        bg_centroids = all_centroids_um[np.random.rand(all_centroids_um.shape[0]) < 0.1, :]

        acl.plot_anatomy(lcoors_sensory[5], bg_centroids, "Sensory", plot_dir, "C1")
        acl.plot_anatomy(lcoors_motor[5], bg_centroids, "Motor", plot_dir, "C2")
        acl.plot_anatomy(lcoors_interact[5], bg_centroids, "Mixed", plot_dir, "C0")

        acl.plot_anatomy(lcoors_comp0[5], bg_centroids, "Comp0", plot_dir, "C0")
        acl.plot_anatomy(lcoors_comp1[5], bg_centroids, "Comp1", plot_dir, "C0")
        acl.plot_anatomy(lcoors_comp2[5], bg_centroids, "Comp2", plot_dir, "C0")

        acl.plot_anatomy(lcoors_start[5], bg_centroids, "Start", plot_dir, "C0")
        acl.plot_anatomy(lcoors_disp[5], bg_centroids, "Disp", plot_dir, "C0")
        acl.plot_anatomy(lcoors_ang[5], bg_centroids, "Dir", plot_dir, "C0")

        # Plot response of neurons that are mixed for bout-start, nonlinear and sensory
        # w.r.t. input drive along sensory and bout start kernels - use clustering to group responses
        # Note: The clustering below is somewhat loose - we use cross-correlations to cluster since we do not
        # expect to have the exact same locations but rather want to cluster on shape of the response landscape
        # However, that term isn't as well defined as we would like
        mixed_bs_sens_nl_clust = np.logical_and(np.logical_and(sens_contrib, non_linear), barcodes[:, start_id] == 1)
        mixed_neuron_ids = np.hstack(df_fit["neuron_id"][mixed_bs_sens_nl_clust])

        # find all drive responses - note this is timetaking
        all_drive_responses = []
        all_dr_fft = []
        similarity_matrix = np.zeros((mixed_neuron_ids.size, mixed_neuron_ids.size))
        for i, ntt in enumerate(mixed_neuron_ids):
            neuron_model = get_neuron_model(ntt, df_neuron_list, df_experiment_list, model_file_folder)
            do = get_drive_outputs(neuron_model, all_jacs[all_fit_neuron_ids == ntt], 0, 1, -10, 10, 100)
            all_drive_responses.append(do)
            print(f"Drive output {i+1} out of {mixed_neuron_ids.size} completed")

        # compute and store fourier transforms to speed up crosscorrelations
        for i, dr in enumerate(all_drive_responses):
            norm = np.linalg.norm(dr)
            all_dr_fft.append(np.fft.fft2(dr) / norm)
            print(f"Fourier transform {i + 1} out of {mixed_neuron_ids.size} completed")

        # use cross-correlations to compute similarities so that we tolerate shifts
        for i in range(mixed_neuron_ids.size):
            for j in range(i, mixed_neuron_ids.size):
                # note: the ordering of elements will be different than if calling scipy.signal.correlate2d
                cross_corr = np.absolute(np.fft.ifft2(all_dr_fft[i] * all_dr_fft[j].conjugate()))
                similarity = np.max(cross_corr)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
                print(f"Cross correlation {i*mixed_neuron_ids.size + j + 1} out of {mixed_neuron_ids.size**2} completed")

        # use spectral clustering to group neurons
        spc = SpectralClustering(n_clusters=10, affinity="precomputed")
        spc.fit(similarity_matrix)
        cid = spc.labels_.copy()

        # plot the first member of each cluster as an exemplar
        for i in range(10):
            to_plot = np.where(cid == i)[0][0]
            fig = plot_drive_response(all_drive_responses[to_plot])
            fig.savefig(path.join(plot_dir, f"Motor_drive_exemplar_C{i}_0.{ext}"), dpi=300)

        # plot distribution of linear model scores for the different types
        fig = pl.figure()
        sns.kdeplot(l_approx_score[interact_cluster], label="Mixed", cut=0)
        sns.kdeplot(l_approx_score[sensory_cluster], label="Stimulus", cut=0)
        sns.kdeplot(l_approx_score[motor_cluster], label="Behavior", cut=0)
        pl.plot([lscore_th, lscore_th], [0, 1], 'k--')
        pl.xlabel("Linear model approximation $R^2$")
        pl.ylabel("Density")
        sns.despine(fig)
        pl.legend()
        fig.savefig(path.join(plot_dir, f"LinModelApprox_Distribution_By_Type.{ext}"), dpi=300)

        # for different clusters compute the fraction of units also identified by the linear model
        fit_lm_fit = np.hstack(df_fit["is_lm_fit"])
        expts = np.unique(exp_fit)
        d_lm = {"Type": [], "LM ID fraction": []}
        for eid in expts:
            this_exp = exp_fit == eid
            # for this experiment get whether ann id units were also lm id'd
            lm_abt = fit_lm_fit[this_exp]
            this_nonlinear = non_linear[this_exp]
            this_sens = sensory_cluster[this_exp]
            this_mot = motor_cluster[this_exp]
            this_int = interact_cluster[this_exp]
            this_complex_2 = complex_2[this_exp]

            d_lm["Type"].append("All")
            d_lm["LM ID fraction"].append(np.sum(lm_abt) / lm_abt.size)

            d_lm["Type"].append("Nonlinear")
            d_lm["LM ID fraction"].append(np.sum(lm_abt[this_nonlinear]) / np.sum(this_nonlinear))

            d_lm["Type"].append("Complexity 2")
            d_lm["LM ID fraction"].append(np.sum(lm_abt[this_complex_2]) / np.sum(this_complex_2))

            d_lm["Type"].append("Sensory")
            d_lm["LM ID fraction"].append(np.sum(lm_abt[this_sens]) / np.sum(this_sens))

            d_lm["Type"].append("Motor")
            d_lm["LM ID fraction"].append(np.sum(lm_abt[this_mot]) / np.sum(this_mot))

            d_lm["Type"].append("Interaction")
            d_lm["LM ID fraction"].append(np.sum(lm_abt[this_int]) / np.sum(this_int))

        df_lm = DataFrame(d_lm)

        fig = pl.figure()
        sns.barplot(x="Type", y="LM ID fraction", data=df_lm, ci=68)
        sns.despine()
        fig.savefig(path.join(plot_dir, f"LM_Identified_Fractions.{ext}"), dpi=300)

        # plot relationship of nonlinearity probability and me-score just for sensory units
        mes_sens = me_score[sensory_cluster]
        lap_sens = l_approx_score[sensory_cluster]
        q1 = np.logical_and(lap_sens < lscore_th, mes_sens >= me_threshold)
        q2 = np.logical_and(lap_sens < lscore_th, mes_sens < me_threshold)
        q3 = np.logical_and(lap_sens >= lscore_th, mes_sens >= me_threshold)
        q4 = np.logical_and(lap_sens >= lscore_th, mes_sens < me_threshold)
        fig = pl.figure()
        pl.scatter(lap_sens[np.logical_or(q3, q4)], mes_sens[np.logical_or(q3, q4)], s=2, alpha=0.25, c='k')
        pl.scatter(lap_sens[q1], mes_sens[q1], s=2, alpha=0.25, c='C0')
        pl.scatter(lap_sens[q2], mes_sens[q2], s=2, alpha=0.25, c='C3')
        pl.plot([0, 1], [me_threshold, me_threshold], 'k--')
        pl.plot([lscore_th, lscore_th], [0, 1], 'k--')
        pl.text(0.25, 0.75, np.sum(q1))
        pl.text(0.25, 0.25, np.sum(q2))
        pl.text(0.75, 0.75, np.sum(q3))
        pl.text(0.75, 0.25, np.sum(q4))
        pl.xlim(-0.01, 1.01)
        pl.ylim(-0.01, 1.01)
        pl.xlabel("Linear model approximation $R^2$")
        pl.ylabel("2nd order model $R^2$")
        sns.despine()
        fig.savefig(path.join(plot_dir, f"mescore_vs_linapproxscore_sensory_units.{ext}"), dpi=300)

        #######################################
        # Plot per-experiment neurons to check
        # registration accuracy and coverage
        #######################################
        validation_dir = path.join(plot_dir, "validation")
        if not path.exists(validation_dir):
            os.makedirs(validation_dir)
        valid = np.logical_not(np.isnan(all_test_corrs))
        all_centroids = np.vstack(df_neuron_list["zbrain_coords"])
        centroids_bg: np.ndarray = all_centroids[valid, :]
        exp_id_bg = np.hstack(df_neuron_list["experiment_id"])[valid]
        sel_random = np.random.rand(centroids_bg.shape[0]) < 0.05
        for i in range(len(experiment_names)):
            fig = pl.figure()
            pl.scatter(centroids_bg[sel_random, 1], centroids_bg[sel_random, 2], s=2, alpha=0.1, color='k')
            pl.scatter(centroids_bg[exp_id_bg == i, 1], centroids_bg[exp_id_bg == i, 2], s=2, alpha=0.1, color='C1')
            pl.title(f"Centroids of experiment {experiment_names[i]}")
            pl.gca().invert_yaxis()
            pl.axis("equal")
            fig.savefig(path.join(validation_dir, f"{experiment_names[i]}_Neurons_Side.png"), dpi=300)
            pl.close(fig)

            fig = pl.figure()
            pl.scatter(centroids_bg[sel_random, 0], centroids_bg[sel_random, 1], s=2, alpha=0.1, color='k')
            pl.scatter(centroids_bg[exp_id_bg == i, 0], centroids_bg[exp_id_bg == i, 1], s=2, alpha=0.1, color='C1')
            pl.title(f"Centroids of experiment {experiment_names[i]}")
            pl.axis("equal")
            fig.savefig(path.join(validation_dir, f"{experiment_names[i]}_Neurons_Top.png"), dpi=300)
            pl.close(fig)

        #######################################
        # Z-Brain anatomy
        #######################################
        # define those regions we want to analyze - striking a compromise between capturing detail and
        # avoiding unneccesary subdivisions
        df_zb_regions = rbm.get_zbrain_region_list(dfile)
        region_names = np.vstack(df_zb_regions["region_name"])
        region_names = [n[0].decode('utf-8') for n in region_names]
        df_zb_anat = rbm.get_zbrain_anatomy(dfile)
        region_mship = np.vstack(df_zb_anat["region_membership"])
        all_centroids_px = np.vstack(df_neuron_list["zbrain_pixels"])

        # Make overview bar-plots that show the fraction of specific types in selected regions relative to the
        # total number of neurons that were fit by the ANN
        regions_for_overview = ['Diencephalon - Caudal Hypothalamus',  # CHYP
                                'Diencephalon - Dorsal Thalamus',  # DTHL
                                'Diencephalon - Eminentia Thalami',  # ETHL
                                'Diencephalon - Habenula',  # HABN
                                'Diencephalon - Intermediate Hypothalamus',  # IHYP
                                'Diencephalon - Migrated Area of the Pretectum (M1)',  # PTM1
                                'Diencephalon - Migrated Posterior Tubercular Area (M2)',  # PTM2
                                'Diencephalon - Pineal',  # PINE
                                'Diencephalon - Pituitary',  # PITU
                                'Diencephalon - Posterior Tuberculum',  # PTUB
                                'Diencephalon - Postoptic Commissure',  # POCM
                                'Diencephalon - Preoptic Area',  # POA
                                'Diencephalon - Pretectum',  # PTCT
                                'Diencephalon - Rostral Hypothalamus',  # RHYP
                                'Diencephalon - Torus Lateralis',  # TOLT
                                'Diencephalon - Ventral Thalamus',  # VTHL
                                'Mesencephalon - Medial Tectal Band',  # MTCB
                                'Mesencephalon - NucMLF (nucleus of the medial longitudinal fascicle)',  # NMLF
                                'Mesencephalon - Oculomotor Nucleus nIII',  # OCN3
                                'Mesencephalon - Tectum Stratum Periventriculare',  # TSPV
                                'Mesencephalon - Tegmentum',  # TEGM
                                'Mesencephalon - Torus Longitudinalis',  # TOLO
                                'Mesencephalon - Torus Semicircularis',  # TOSC
                                'Rhombencephalon - Anterior Cluster of nV Trigeminal Motorneurons',  # AVTM
                                'Rhombencephalon - Area Postrema',  # ARPS
                                'Rhombencephalon - CaD',  # RCAD
                                'Rhombencephalon - CaV',  # RCAV
                                'Rhombencephalon - Cerebellum',  # CRBL
                                'Rhombencephalon - Corpus Cerebelli',  # CCRB
                                'Rhombencephalon - Eminentia Granularis',  # EMGR
                                'Rhombencephalon - Gad1b Stripe 1',  # GAD1
                                'Rhombencephalon - Gad1b Stripe 2',  # GAD2
                                'Rhombencephalon - Gad1b Stripe 3',  # GAD3
                                'Rhombencephalon - Glyt2 Stripe 1',  # GLY1
                                'Rhombencephalon - Glyt2 Stripe 2',  # GLY2
                                'Rhombencephalon - Glyt2 Stripe 3',  # GLY3
                                'Rhombencephalon - Inferior Olive',  # INFO
                                'Rhombencephalon - Interpeduncular Nucleus',  # IPN
                                'Rhombencephalon - Isl1 Stripe 1',  # ISL1
                                'Rhombencephalon - Lateral Reticular Nucleus',  # LREN
                                'Rhombencephalon - Lobus caudalis cerebelli',  # LBCC
                                'Rhombencephalon - Locus Coreuleus',  # LCOE
                                'Rhombencephalon - Medial Vestibular Nucleus',  # MVNC
                                'Rhombencephalon - Noradrendergic neurons of the Interfascicular and Vagal areas',
                                # NNIV
                                'Rhombencephalon - Oculomotor Nucleus nIV',  # OCN4
                                'Rhombencephalon - Olig2 Stripe',  # OLI2
                                'Rhombencephalon - Posterior Cluster of nV Trigeminal Motorneurons',  # PVTM
                                'Rhombencephalon - Raphe - Inferior',  # RPIN
                                'Rhombencephalon - Raphe - Superior',  # RPSP
                                'Rhombencephalon - Rhombomere 1',  # RH1
                                'Rhombencephalon - Rhombomere 2',  # RH2
                                'Rhombencephalon - Rhombomere 3',  # RH3
                                'Rhombencephalon - Rhombomere 4',  # RH4
                                'Rhombencephalon - Rhombomere 5',  # RH5
                                'Rhombencephalon - Rhombomere 6',  # RH6
                                'Rhombencephalon - Rhombomere 7',  # RH7
                                'Rhombencephalon - VII Facial Motor and octavolateralis efferent neurons',  # VII
                                "Rhombencephalon - VII' Facial Motor and octavolateralis efferent neurons",  # VIIP
                                'Rhombencephalon - Valvula Cerebelli',  # VCRB
                                'Rhombencephalon - Vglut2 Stripe 1',  # VGL1
                                'Rhombencephalon - Vglut2 Stripe 2',  # VGL2
                                'Rhombencephalon - Vglut2 Stripe 3',  # VGL3
                                'Rhombencephalon - Vglut2 Stripe 4',  # VGL4
                                'Rhombencephalon - Vmat2 Stripe1',  # VMT1
                                'Rhombencephalon - Vmat2 Stripe2',  # VMT2
                                'Rhombencephalon - Vmat2 Stripe3',  # VMT3
                                'Rhombencephalon - X Vagus motorneuron cluster',  # XVMN
                                'Telencephalon - Anterior Commisure',  # AC
                                'Telencephalon - Optic Commissure',  # OC
                                'Telencephalon - Pallium',  # PALL
                                'Telencephalon - Postoptic Commissure',  # POC
                                'Telencephalon - Subpallial Otpb strip',  # SPOS
                                'Telencephalon - Subpallium',  # SPAL
                                'Telencephalon - Telencephalic Migrated Area 4 (M4)'  # TEM4
                                ]
        rgnn_abbreviations = {'Diencephalon - Caudal Hypothalamus': "CHYP",
                              'Diencephalon - Dorsal Thalamus': "DTHL",
                              'Diencephalon - Eminentia Thalami': "ETHL",
                              'Diencephalon - Habenula': "HABN",
                              'Diencephalon - Intermediate Hypothalamus': "IHYP",
                              'Diencephalon - Migrated Area of the Pretectum (M1)': "PTM1",
                              'Diencephalon - Migrated Posterior Tubercular Area (M2)': "PTM2",
                              'Diencephalon - Pineal': "PINE",
                              'Diencephalon - Pituitary': "PITU",
                              'Diencephalon - Posterior Tuberculum': "PTUB",
                              'Diencephalon - Postoptic Commissure': "POCM",
                              'Diencephalon - Preoptic Area': "POA",
                              'Diencephalon - Pretectum': "PTCT",
                              'Diencephalon - Rostral Hypothalamus': "RHYP",
                              'Diencephalon - Torus Lateralis': "TOLT",
                              'Diencephalon - Ventral Thalamus': "VTHL",
                              'Mesencephalon - Medial Tectal Band': "MTCB",
                              'Mesencephalon - NucMLF (nucleus of the medial longitudinal fascicle)': "NMLF",
                              'Mesencephalon - Oculomotor Nucleus nIII': "OCN3",
                              'Mesencephalon - Tectum Stratum Periventriculare': "TSPV",
                              'Mesencephalon - Tegmentum': "TEGM",
                              'Mesencephalon - Torus Longitudinalis': "TOLO",
                              'Mesencephalon - Torus Semicircularis': "TOSC",
                              'Rhombencephalon - Anterior Cluster of nV Trigeminal Motorneurons': "AVTM",
                              'Rhombencephalon - Area Postrema': "ARPS",
                              'Rhombencephalon - CaD': "RCAD",
                              'Rhombencephalon - CaV': "RCAV",
                              'Rhombencephalon - Cerebellum': "CRBL",
                              'Rhombencephalon - Corpus Cerebelli': "CCRB",
                              'Rhombencephalon - Eminentia Granularis': "EMGR",
                              'Rhombencephalon - Gad1b Stripe 1': "GAD1",
                              'Rhombencephalon - Gad1b Stripe 2': "GAD2",
                              'Rhombencephalon - Gad1b Stripe 3': "GAD3",
                              'Rhombencephalon - Glyt2 Stripe 1': "GLY1",
                              'Rhombencephalon - Glyt2 Stripe 2': "GLY2",
                              'Rhombencephalon - Glyt2 Stripe 3': "GLY3",
                              'Rhombencephalon - Inferior Olive': "INFO",
                              'Rhombencephalon - Interpeduncular Nucleus': "IPN",
                              'Rhombencephalon - Isl1 Stripe 1': "ISL1",
                              'Rhombencephalon - Lateral Reticular Nucleus': "LREN",
                              'Rhombencephalon - Lobus caudalis cerebelli': "LBCC",
                              'Rhombencephalon - Locus Coreuleus': "LCOE",
                              'Rhombencephalon - Medial Vestibular Nucleus': "MVNC",
                              'Rhombencephalon - Noradrendergic neurons of the Interfascicular and Vagal areas': "NNIV",
                              'Rhombencephalon - Oculomotor Nucleus nIV': "OCN4",
                              'Rhombencephalon - Olig2 Stripe': "OLI2",
                              'Rhombencephalon - Posterior Cluster of nV Trigeminal Motorneurons': "PVTM",
                              'Rhombencephalon - Raphe - Inferior': "RPIN",
                              'Rhombencephalon - Raphe - Superior': "RPSP",
                              'Rhombencephalon - Rhombomere 1': "RH1",
                              'Rhombencephalon - Rhombomere 2': "RH2",
                              'Rhombencephalon - Rhombomere 3': "RH3",
                              'Rhombencephalon - Rhombomere 4': "RH4",
                              'Rhombencephalon - Rhombomere 5': "RH5",
                              'Rhombencephalon - Rhombomere 6': "RH6",
                              'Rhombencephalon - Rhombomere 7': "RH7",
                              'Rhombencephalon - VII Facial Motor and octavolateralis efferent neurons': "VII",
                              "Rhombencephalon - VII' Facial Motor and octavolateralis efferent neurons": "VIIP",
                              'Rhombencephalon - Valvula Cerebelli': "VCRB",
                              'Rhombencephalon - Vglut2 Stripe 1': "VGL1",
                              'Rhombencephalon - Vglut2 Stripe 2': "VGL2",
                              'Rhombencephalon - Vglut2 Stripe 3': "VGL3",
                              'Rhombencephalon - Vglut2 Stripe 4': "VGL4",
                              'Rhombencephalon - Vmat2 Stripe1': "VMT1",
                              'Rhombencephalon - Vmat2 Stripe2': "VMT2",
                              'Rhombencephalon - Vmat2 Stripe3': "VMT3",
                              'Rhombencephalon - X Vagus motorneuron cluster': "XVMN",
                              'Telencephalon - Anterior Commisure': "AC",
                              'Telencephalon - Optic Commissure': "OC",
                              'Telencephalon - Pallium': "PALL",
                              'Telencephalon - Postoptic Commissure': "POC",
                              'Telencephalon - Subpallial Otpb strip': "SPOS",
                              'Telencephalon - Subpallium': "SPAL",
                              'Telencephalon - Telencephalic Migrated Area 4 (M4)': "TEM4"
                              }
        # get the indices that correspond to our list of regions to for overivew
        ix_overview = [region_names.index(n) for n in regions_for_overview]
        mask_file = h5py.File("ZBrain_Masks.hdf5", 'r')
        mask_outlines = mask_file["Outlines"]
        brain_outline_top = np.sum(mask_outlines["Rhombencephalon -"][()] + mask_outlines["Telencephalon -"][()] +
                                   mask_outlines["Diencephalon -"][()] + mask_outlines["Mesencephalon -"][()], 2) * 5
        brain_outline_top[brain_outline_top > 50] = 50
        brain_outline_side = np.sum(mask_outlines["Rhombencephalon -"][()] + mask_outlines["Telencephalon -"][()] +
                                    mask_outlines["Diencephalon -"][()] + mask_outlines["Mesencephalon -"][()], 1) * 5
        brain_outline_side[brain_outline_side > 200] = 200
        plot_region_enrichment(sensory_cluster, above_thresh[above_thresh], mask_outlines, brain_outline_top,
                               brain_outline_side, "Sensory", regions_for_overview,
                               region_mship[above_thresh][:, ix_overview])
        plot_region_enrichment(motor_cluster, above_thresh[above_thresh], mask_outlines, brain_outline_top,
                               brain_outline_side, "Motor", regions_for_overview,
                               region_mship[above_thresh][:, ix_overview])
        plot_region_enrichment(interact_cluster, above_thresh[above_thresh], mask_outlines, brain_outline_top,
                               brain_outline_side, "Mixed", regions_for_overview,
                               region_mship[above_thresh][:, ix_overview])

        # For a shared region plot actual neurons mixed vs. sensory
        ix_tma4 = region_names.index('Telencephalon - Telencephalic Migrated Area 4 (M4)')
        abt_in_tma4 = region_mship[above_thresh, ix_tma4]
        fig = pl.figure()
        pl.imshow(np.sum(mask_outlines["Telencephalon -"], 2), cmap='bone')
        pl.imshow(np.sum(mask_outlines["Telencephalon - Telencephalic Migrated Area 4 (M4)"], 2), cmap='hot', alpha=0.5)
        pl.scatter(all_centroids_px[above_thresh][np.logical_and(sensory_cluster, abt_in_tma4), 1],
                   all_centroids_px[above_thresh][np.logical_and(sensory_cluster, abt_in_tma4), 0], s=2)
        pl.scatter(all_centroids_px[above_thresh][np.logical_and(interact_cluster, abt_in_tma4), 1],
                   all_centroids_px[above_thresh][np.logical_and(interact_cluster, abt_in_tma4), 0], s=2)
        fig.savefig(path.join(plot_dir, f"Neuron_Loc_MixAndSens_TelMigArea4.{ext}"), dpi=900)

        # Plot stimulus neurons in olfactory epithelium
        ix_olf = region_names.index('Ganglia - Olfactory Epithelium')
        abt_in_olf = region_mship[above_thresh, ix_olf]
        n_abt_in_olf = region_mship[np.logical_not(above_thresh), ix_olf]
        fig = pl.figure()
        pl.imshow(np.sum(mask_outlines["Telencephalon -"], 2), cmap='bone')
        pl.imshow(np.sum(mask_outlines["Ganglia - Olfactory Epithelium"], 2), cmap='bone', alpha=0.5)
        pl.scatter(all_centroids_px[np.logical_not(above_thresh)][n_abt_in_olf, 1],
                   all_centroids_px[np.logical_not(above_thresh)][n_abt_in_olf, 0], s=1, alpha=0.2)
        pl.scatter(all_centroids_px[above_thresh][np.logical_and(sensory_cluster, abt_in_olf), 1],
                   all_centroids_px[above_thresh][np.logical_and(sensory_cluster, abt_in_olf), 0], s=2)
        pl.axis('equal')
        fig.savefig(path.join(plot_dir, f"Stim_Neuron_Loc_OlfEpith.{ext}"), dpi=900)

        # Plot enriched regions for motor types
        motor_assigned = np.logical_or(start_cluster, np.logical_or(ang_cluster, disp_cluster))
        plot_region_enrichment(start_cluster, motor_assigned, mask_outlines, brain_outline_top, brain_outline_side,
                               "swim_start", regions_for_overview, region_mship[above_thresh][:, ix_overview])
        plot_region_enrichment(disp_cluster, motor_assigned, mask_outlines, brain_outline_top, brain_outline_side,
                               "displacement", regions_for_overview, region_mship[above_thresh][:, ix_overview])
        plot_region_enrichment(ang_cluster, motor_assigned, mask_outlines, brain_outline_top, brain_outline_side,
                               "turn_angle", regions_for_overview, region_mship[above_thresh][:, ix_overview])

        # plot temperature representation in olfactory system

        #######################################
        # Subclustering of sensory units by
        # receptive field and complexity
        #######################################
        complexity_sens = complexity[sensory_cluster].astype(float)
        jac_cluster = np.hstack(df_fit["jac_cluster"])
        jac_cluster_sens = jac_cluster[sensory_cluster]
        assert np.min(jac_cluster_sens) == -1  # -2 should be reserved for non-sensory
        jac_sensory = np.vstack(df_fit["jacobian"])[sensory_cluster][:, :50]

        # plot receptive fields of all clusters together with counts
        n_r = 3
        n_c = 5
        fig, axes = pl.subplots(nrows=n_r, ncols=n_c, figsize=[13, 4.8])
        time = (-np.arange(50) / 5)[::-1]
        for r in range(n_r):
            for c in range(n_c):
                cnum = r + n_r * c
                axis = axes[r, c]
                rfield = np.mean(jac_sensory[jac_cluster_sens == cnum], 0)
                rfield /= np.linalg.norm(rfield)
                axis.plot(time, rfield, label=f"{cnum}: {np.sum(jac_cluster_sens == cnum)}")
                axis.legend()
                if r == n_r - 1:
                    axis.set_xlabel("Time [s]")
                if c == 0:
                    axis.set_ylabel("dCa/dt")
        sns.despine()
        fig.tight_layout()
        fig.savefig(path.join(plot_dir, f"SensoryJacobians_Clustered.{ext}"))

        # plot heatmap showing for each jacobian cluster which fraction of it is in the four major brain
        # regions as well as heatmap of the actual receptive fields to be joined later into one plot
        major_region_indices = [
            region_names.index("Telencephalon -"),
            region_names.index("Diencephalon -"),
            region_names.index("Mesencephalon -"),
            region_names.index("Rhombencephalon -")
        ]
        region_mship_sens_major = region_mship[above_thresh][sensory_cluster][:, major_region_indices].astype(float)
        sjac_brain_distrib = np.empty((15, 4))
        sjac_recfields = np.empty((15, 50))
        for cnum in range(15):
            cluster_count = np.sum(jac_cluster_sens == cnum)
            rfield = np.mean(jac_sensory[jac_cluster_sens == cnum], 0)
            rfield /= np.linalg.norm(rfield)
            sjac_recfields[cnum, :] = rfield
            sjac_brain_distrib[cnum, :] = np.sum(region_mship_sens_major[jac_cluster_sens == cnum], 0) / cluster_count
        fig = pl.figure()
        sns.heatmap(sjac_recfields, center=0, xticklabels=[], cmap='vlag')
        pl.ylabel("Cluster")
        pl.xlabel("Time [s]")
        fig.savefig(path.join(plot_dir, f"SensoryJacobians_ClusterHeatmap.{ext}"))
        fig = pl.figure()
        sns.heatmap(sjac_brain_distrib, center=0, xticklabels=["Telencephalon", "Diencephalon", "Mesencephalon",
                                                               "Rhombencephalon"])
        pl.ylabel("Cluster")
        fig.tight_layout()
        fig.savefig(path.join(plot_dir, f"SensoryJacobians_BrainDistrib.{ext}"))

        # plot average complexity in each given cluster together with error metric around expectation
        avg_complexity = np.mean(complexity_sens)
        lower = []
        upper = []
        for cnum in range(15):
            l, u = boot_complexity(complexity_sens, np.sum(jac_cluster_sens == cnum)[()], 95.0, 1000)[:2]
            lower.append(l)
            upper.append(u)
        figure = pl.figure()
        sns.barplot(y=[np.mean(complexity_sens[jac_cluster_sens == cnum]) for cnum in range(15)], x=np.arange(15),
                    color='grey')
        sns.pointplot(y=lower, x=np.arange(15), color='C0', alpha=0.5, markers="+", join=False)
        sns.pointplot(y=np.full(15, avg_complexity), x=np.arange(15), color='C0', markers="None")
        sns.pointplot(y=upper, x=np.arange(15), color='C0', alpha=0.5, markers="+", join=False)
        sns.despine()
        pl.xlabel("Receptive field cluster")
        pl.ylabel("Average complexity score")
        figure.tight_layout()
        figure.savefig(path.join(plot_dir, f"JacCluster_AvgComplexity.{ext}"), dpi=300)
