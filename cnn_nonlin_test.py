"""
Script to test nonlinearity metrics
"""

from scipy.stats.mstats import mquantiles
import model
import numpy as np
import matplotlib.pyplot as pl
import utilities
import seaborn as sns
import time
from taylorDecomp import d2ca_dr2, complexity_scores, avg_directional_curvature
import matplotlib as mpl
import os
from os import path
from sklearn.metrics import auc
from typing import Tuple
import h5py
from perf_nlc_nonlin import calc_nlc


def standardize(ar):
    return (ar - np.mean(ar)) / np.std(ar)


def diff_predictor(predictor: np.ndarray, d_len=10) -> np.ndarray:
    if d_len < 2:
        raise ValueError("d_len has to be 2 or larger")
    f = np.zeros(d_len)
    f[:d_len//2] = -1
    f[d_len//2:] = 1
    return np.convolve(predictor, f)[:predictor.size]


def convolve_predictor(predictor: np.ndarray, tau_0: float, tau_1: float, scale: float) ->\
        Tuple[np.ndarray, np.ndarray]:
    if tau_0 <= 0 or tau_1 <= 0 or scale < 0:
        raise ValueError("tau_0, tau_1 and scale have to be > 0")
    t_base = np.linspace(0, 10, 50)
    f = scale * t_base * np.exp(-t_base / tau_0) + (1-t_base)*np.exp(-t_base / tau_1)
    return np.convolve(predictor, f, 'full')[:predictor.size], f


def rect_predictor(predictor: np.ndarray, scale=1.0, offset=1.0) -> np.ndarray:
    if offset < 1.0:
        raise ValueError("offset has to be 1 or larger")  # while >=0 is valid, it would be a linear function
    if scale < 1.0:
        raise ValueError("Scale has to be >= 1")
    return np.log(np.exp(scale*predictor) + offset)  # softplus


def tanhsq_predictor(predictor: np.ndarray, exponent: int = 2) -> np.ndarray:
    if exponent < 1:
        raise ValueError("exponent has to be integer >= 1")
    return np.tanh(2*predictor) ** exponent


def roc_analysis(linear: np.ndarray, non_linear: np.ndarray, thresh_to_test: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Compute true and false positive rates on known data based on a set of thresholds
    :param linear: Metrics belonging to the known linear class
    :param non_linear: Metrics belonging to the known non-linear class
    :param thresh_to_test: The metric thresholds to use
    :return:
        [0]: True positive rates for each threshold
        [1]: False positive rates for each threshold
    """
    valid_lin = np.logical_not(np.isnan(linear))
    valid_nlin = np.logical_not(np.isnan(non_linear))
    linear = linear[valid_lin]
    non_linear = non_linear[valid_nlin]
    false_positive_rate = []
    true_positive_rate = []
    for th in thresh_to_test:
        # NOTE: Our scores are a linearity metric not a non-linearity metric therefore <threshold comparison
        true_pos = np.sum(non_linear < th)
        true_positive_rate.append(true_pos / non_linear.size)
        false_pos = np.sum(linear < th)
        false_positive_rate.append(false_pos / linear.size)
    return np.hstack(true_positive_rate), np.hstack(false_positive_rate)


if __name__ == '__main__':
    # the following will prevent tensorflow from using the GPU - as the used models have very low complexity
    # they will generally be fit faster on the CPU - comment out to run on the GPU instead
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    mpl.rcParams['pdf.fonttype'] = 42

    plot_dir = "cnn_nonlin_test_plots"
    if not path.exists(plot_dir):
        os.makedirs(plot_dir)

    frame_rate = 5
    time_base = np.arange(1500*5) / frame_rate
    assert time_base.size % 3 == 0
    hist_seconds = 10
    hist_steps = hist_seconds * frame_rate
    hist_frames = hist_seconds * frame_rate
    train_frames = (time_base.size * 2) // 3
    n_epochs = 100

    # granularity of complexity and curvature determination
    analyze_every = frame_rate*5  # compute every five seconds
    look_ahead = frame_rate * 5  # use 5s look-ahead to compute input data direction

    # all signals are standardized to unit variance/sd - noise_fraction controls the standard deviation of the noise
    noise_fraction = 0.25

    # To keep results consistent for testing purposes
    np.random.seed(777)

    reg = utilities.create_random_wave_predictor(time_base)
    diffed = diff_predictor(reg)
    diffed -= np.mean(diffed)
    diffed /= np.std(diffed)
    regressors = [reg]
    # We will build responses from linear -> nonlinear by blending reg with power transformations of reg
    rectified = rect_predictor(reg, scale=3.0)
    rectified /= np.std(rectified)
    tanhsq = tanhsq_predictor(reg)
    tanhsq /= np.std(tanhsq)

    ca_responses = []

    for i in range(10):
        r = i / 9 * rectified + (9 - i) / 9 * reg
        ca_responses.append(r)

    for i in range(10):
        r = i / 9 * tanhsq + (9 - i) / 9 * reg
        ca_responses.append(r)

    for i in range(10):
        r = i/9 * diffed + (9-i)/9 * reg
        ca_responses.append(r)

    ca_responses = np.vstack(ca_responses)
    ca_responses -= np.mean(ca_responses, 1, keepdims=True)
    ca_responses /= np.std(ca_responses, 1, keepdims=True) + 1e-9
    # add noise
    ca_responses += np.random.randn(ca_responses.shape[0], ca_responses.shape[1]) * noise_fraction

    pl.figure()
    sns.heatmap(ca_responses, rasterized=True)
    sns.despine()
    pl.title("Ca responses")

    data = utilities.Data(hist_steps, regressors, ca_responses, train_frames)
    correlations_trained = []
    correlations_test = []
    all_sq_scores = []  # scores of the 2nd-order model
    all_lin_scores = []  # scores of the 1st-order model

    for cell_ix in range(ca_responses.shape[0]):
        start = time.perf_counter()
        tset = data.training_data(cell_ix, batch_size=256)
        m = model.get_standard_model(hist_steps)
        # the following is required to init variables at desired shape
        m(np.random.randn(1, hist_steps, len(regressors)).astype(np.float32))
        # train
        model.train_model(m, tset, n_epochs, ca_responses.shape[1])
        # evaluate
        p, r = data.predict_response(cell_ix, m)
        c_tr = np.corrcoef(p[:train_frames], r[:train_frames])[0, 1]
        correlations_trained.append(c_tr)
        c_ts = np.corrcoef(p[train_frames:], r[train_frames:])[0, 1]
        correlations_test.append(c_ts)
        stop = time.perf_counter()
        print(f"Model for neuron {cell_ix} achieved correlation {c_tr} on training data and {c_ts} on test data.")
        print(f"Training took {int(stop - start)} seconds")

        regs = data.regressor_matrix(cell_ix)
        start = time.perf_counter()
        # obtain data-mean and J and H at data mean
        all_inputs = []
        for inp, outp in tset:
            all_inputs.append(inp.numpy())
        x_bar = np.mean(np.vstack(all_inputs), 0, keepdims=True)
        jacobian, hessian = d2ca_dr2(m, x_bar)
        # compute complexity scores
        lin_score, sq_score = complexity_scores(m, x_bar, jacobian, hessian, regs, analyze_every)
        # NOTE: Since these are scores of a truncated model they can be < 0 to ease plotting and interpretation
        # we set negative scores to 0 and hence collapse them to "no variance explained" instead of "less variance
        # explained than the average output"
        if lin_score < 0:
            lin_score = 0
        if sq_score < 0:
            sq_score = 0
        all_lin_scores.append(lin_score)
        all_sq_scores.append(sq_score)
        stop = time.perf_counter()
        print(f"Complexity calculation took {int(stop - start)} seconds; Linear model score: {lin_score}")
        print(f"2nd order model score: {sq_score}")
        print()

    all_lin_scores = np.hstack(all_lin_scores)
    all_sq_scores = np.hstack(all_sq_scores)
    correlations_test = np.hstack(correlations_test)

    # mixing effect of rectification
    fig, (ax_corr, ax_curve, ax_nlc) = pl.subplots(ncols=3)
    ax_corr.plot(np.arange(10)/9, correlations_test[:10]**2, 'ko')
    ax_corr.plot(0, correlations_test[0]**2, 'C0o')
    ax_corr.plot(5/9, correlations_test[5]**2, 'C1o')
    ax_corr.plot(1, correlations_test[9]**2, 'C2o')
    ax_corr.set_xlabel("Nonlinear mix")
    ax_corr.set_ylabel("Test R2")
    ax_corr.set_ylim(0, 1)
    ax_curve.plot(np.arange(10)/9, all_lin_scores[:10], 'ko')
    ax_curve.plot(0, all_lin_scores[0], 'C0o')
    ax_curve.plot(5/9, all_lin_scores[5], 'C1o')
    ax_curve.plot(1, all_lin_scores[9], 'C2o')
    ax_curve.set_ylim(0, np.max(all_lin_scores)+0.05)
    ax_curve.set_xlabel("Nonlinear mix")
    ax_curve.set_ylabel("Linear model score")
    ax_nlc.plot(np.arange(10)/9, all_sq_scores[:10], 'ko')
    ax_nlc.plot(0, all_sq_scores[0], 'C0o')
    ax_nlc.plot(5/9, all_sq_scores[5], 'C1o')
    ax_nlc.plot(1, all_sq_scores[9], 'C2o')
    ax_nlc.set_ylim(0, np.max(all_sq_scores)+0.05)
    ax_nlc.set_xlabel("Nonlinear mix")
    ax_nlc.set_ylabel("2nd order model score")
    fig.tight_layout()
    sns.despine(fig)
    fig.savefig(path.join(plot_dir, "RectificationMixing_Metrics.pdf"))

    fig, axes = pl.subplots(ncols=3)
    axes[0].scatter(reg, ca_responses[0], c='C0', s=2)
    axes[0].set_title(f"R2 = {np.round(np.corrcoef(reg, ca_responses[0])[0, 1] ** 2, 2)}")
    axes[1].scatter(reg, ca_responses[5], c='C1', s=2)
    axes[1].set_title(f"R2 = {np.round(np.corrcoef(reg, ca_responses[5])[0, 1] ** 2, 2)}")
    axes[2].scatter(reg, ca_responses[9], c='C2', s=2)
    axes[2].set_title(f"R2 = {np.round(np.corrcoef(reg, ca_responses[9])[0, 1] ** 2, 2)}")
    for a in axes:
        a.set_xlabel("Predictor value")
        a.set_ylabel("Neural activity")
    fig.tight_layout()
    sns.despine(fig)
    fig.savefig(path.join(plot_dir, "RectificationMixing_ExampleCorrelations.pdf"))

    # mixing effect of squared tanh
    fig, (ax_corr, ax_curve, ax_nlc) = pl.subplots(ncols=3)
    ax_corr.plot(np.arange(10)/9, correlations_test[10:20]**2, 'ko')
    ax_corr.plot(0, correlations_test[10]**2, 'C3o')
    ax_corr.plot(5/9, correlations_test[15]**2, 'C4o')
    ax_corr.plot(1, correlations_test[19]**2, 'C5o')
    ax_corr.set_xlabel("Nonlinear mix")
    ax_corr.set_ylabel("Test R2")
    ax_corr.set_ylim(0, 1)
    ax_curve.plot(np.arange(10)/9, all_lin_scores[10:20], 'ko')
    ax_curve.plot(0, all_lin_scores[10], 'C3o')
    ax_curve.plot(5/9, all_lin_scores[15], 'C4o')
    ax_curve.plot(1, all_lin_scores[19], 'C5o')
    ax_curve.set_ylim(0, np.max(all_lin_scores)+0.05)
    ax_curve.set_xlabel("Nonlinear mix")
    ax_curve.set_ylabel("Linear model score")
    ax_nlc.plot(np.arange(10) / 9, all_sq_scores[10:20], 'ko')
    ax_nlc.plot(0, all_sq_scores[10], 'C3o')
    ax_nlc.plot(5 / 9, all_sq_scores[15], 'C4o')
    ax_nlc.plot(1, all_sq_scores[19], 'C5o')
    ax_nlc.set_ylim(0, np.max(all_sq_scores) + 0.05)
    ax_nlc.set_xlabel("Nonlinear mix")
    ax_nlc.set_ylabel("2nd order model score")
    fig.tight_layout()
    sns.despine(fig)
    fig.savefig(path.join(plot_dir, "TanhSqMixing_Metrics.pdf"))

    fig, axes = pl.subplots(ncols=3)
    axes[0].scatter(reg, ca_responses[10], c='C3', s=2)
    axes[0].set_title(f"R2 = {np.round(np.corrcoef(reg, ca_responses[10])[0, 1]**2, 2)}")
    axes[1].scatter(reg, ca_responses[15], c='C4', s=2)
    axes[1].set_title(f"R2 = {np.round(np.corrcoef(reg, ca_responses[15])[0, 1]**2, 2)}")
    axes[2].scatter(reg, ca_responses[19], c='C5', s=2)
    axes[2].set_title(f"R2 = {np.round(np.corrcoef(reg, ca_responses[19])[0, 1]**2, 2)}")
    for a in axes:
        a.set_xlabel("Predictor value")
        a.set_ylabel("Neural activity")
    fig.tight_layout()
    sns.despine(fig)
    fig.savefig(path.join(plot_dir, "TanhSqMixing_ExampleCorrelations.pdf"))

    # mixing effect of differentiation - a linear operation
    fig, (ax_corr, ax_curve, ax_nlc) = pl.subplots(ncols=3)
    ax_corr.plot(np.arange(10) / 9, correlations_test[20:] ** 2, 'ko')
    ax_corr.plot(0, correlations_test[20] ** 2, 'C6o')
    ax_corr.plot(5 / 9, correlations_test[25] ** 2, 'C7o')
    ax_corr.plot(1, correlations_test[29] ** 2, 'C8o')
    ax_corr.set_xlabel("Derivative mix")
    ax_corr.set_ylabel("Test R2")
    ax_corr.set_ylim(0, 1)
    ax_curve.plot(np.arange(10) / 9, all_lin_scores[20:], 'ko')
    ax_curve.plot(0, all_lin_scores[20], 'C6o')
    ax_curve.plot(5 / 9, all_lin_scores[25], 'C7o')
    ax_curve.plot(1, all_lin_scores[29], 'C8o')
    ax_curve.set_ylim(0, np.max(all_lin_scores)+0.05)
    ax_curve.set_xlabel("Derivative mix")
    ax_curve.set_ylabel("Linear model score")
    ax_nlc.plot(np.arange(10) / 9, all_sq_scores[20:], 'ko')
    ax_nlc.plot(0, all_sq_scores[20], 'C6o')
    ax_nlc.plot(5 / 9, all_sq_scores[25], 'C7o')
    ax_nlc.plot(1, all_sq_scores[29], 'C8o')
    ax_nlc.set_ylim(0, np.max(all_sq_scores) + 0.05)
    ax_nlc.set_xlabel("Derivative mix")
    ax_nlc.set_ylabel("2nd order model score")
    fig.tight_layout()
    sns.despine(fig)
    fig.savefig(path.join(plot_dir, "DerivativeMixing_Metrics.pdf"))

    fig, axes = pl.subplots(ncols=3)
    axes[0].scatter(reg, ca_responses[20], c='C6', s=2)
    axes[0].set_title(f"R2 = {np.round(np.corrcoef(reg, ca_responses[20])[0, 1] ** 2, 2)}")
    axes[1].scatter(reg, ca_responses[25], c='C7', s=2)
    axes[1].set_title(f"R2 = {np.round(np.corrcoef(reg, ca_responses[25])[0, 1] ** 2, 2)}")
    axes[2].scatter(reg, ca_responses[29], c='C8', s=2)
    axes[2].set_title(f"R2 = {np.round(np.corrcoef(reg, ca_responses[29])[0, 1] ** 2, 2)}")
    for a in axes:
        a.set_xlabel("Predictor value")
        a.set_ylabel("Neural activity")
    fig.tight_layout()
    sns.despine(fig)
    fig.savefig(path.join(plot_dir, "DerivativeMixing_ExampleCorrelations.pdf"))

    # test stability of linear model prediction metric across multiple random stimuli
    n_stim = 500
    lscore_diffed = np.full(n_stim, np.nan)
    lscore_rect = np.full(n_stim, np.nan)
    lscore_tanhpow = np.full(n_stim, np.nan)
    lscore_conved = np.full(n_stim, np.nan)

    # the following are for comparative purposees
    all_lscore = []
    all_sqscore = []
    all_crv = []
    all_nlc = []

    all_regressors = []

    all_filters = []

    for i in range(n_stim):
        # create regressor matrix
        reg = standardize(utilities.create_random_wave_predictor(time_base))
        reg2 = standardize(utilities.create_random_wave_predictor(time_base))
        reg3 = standardize(utilities.create_random_wave_predictor(time_base))
        reg4 = standardize(utilities.create_random_wave_predictor(time_base))
        reg5 = standardize(utilities.create_random_wave_predictor(time_base))
        all_regressors.append(reg)
        regressors = [reg, reg2, reg3, reg4, reg5]
        # create responses
        s = np.random.uniform(0.5, 2.0)
        t0 = np.random.uniform(0.5, 2.0)
        t1 = np.random.uniform(0.5, 2.0)
        conved, flter = convolve_predictor(reg, t0, t1, s)
        all_filters.append(flter)
        conved -= np.mean(conved)
        conved /= np.std(conved)
        s = np.random.uniform(2.0, 5.0)
        o = np.random.uniform(1.0, 5.0)
        rectified = rect_predictor(reg, s, o)
        rectified /= np.std(rectified)
        e = np.random.randint(1, 5)
        tanhsq = tanhsq_predictor(reg, e)
        tanhsq /= np.std(tanhsq)
        dif_len = np.random.randint(2, 15)
        diffed = diff_predictor(reg, dif_len)
        diffed -= np.mean(diffed)
        diffed /= np.std(diffed)

        ca_responses = [
            diffed.copy(),
            rectified.copy(),
            tanhsq.copy(),
            conved.copy()
        ]
        ca_responses = np.vstack(ca_responses)
        ca_responses -= np.mean(ca_responses, 1, keepdims=True)
        ca_responses /= np.std(ca_responses, 1, keepdims=True) + 1e-9
        assert np.sum(np.isnan(ca_responses)) == 0
        # add noise
        ca_responses += np.random.randn(ca_responses.shape[0], ca_responses.shape[1]) * noise_fraction
        # create data structure
        data = utilities.Data(hist_steps, regressors, ca_responses, train_frames)
        for cell_ix in range(ca_responses.shape[0]):
            start = time.perf_counter()
            tset = data.training_data(cell_ix, batch_size=256)
            m = model.get_standard_model(hist_steps)
            # the following is required to init variables at desired shape
            m(np.random.randn(1, hist_steps, len(regressors)).astype(np.float32))
            # train
            model.train_model(m, tset, n_epochs, ca_responses.shape[1])
            stop = time.perf_counter()
            print(f"Training took {int(stop - start)} seconds. {i*4+cell_ix} of {n_stim*4} models completed.")
            regs = data.regressor_matrix(cell_ix)
            start = time.perf_counter()
            # obtain data-mean and J and H at data mean
            all_inputs = []
            for inp, outp in tset:
                all_inputs.append(inp.numpy())
            x_bar = np.mean(np.vstack(all_inputs), 0, keepdims=True)
            jacobian, hessian = d2ca_dr2(m, x_bar)
            # compute complexity scores
            lin_score, sq_score = complexity_scores(m, x_bar, jacobian, hessian, regs, analyze_every)
            # NOTE: Since these are scores of a truncated model they can be < 0 to ease plotting and interpretation
            # we set negative scores to 0 and hence collapse them to "no variance explained" instead of "less variance
            # explained than the average output"
            if lin_score < 0:
                lin_score = 0
            if sq_score < 0:
                sq_score = 0
            # store in our all-score variables
            all_lscore.append(lin_score)
            all_sqscore.append(sq_score)
            stop = time.perf_counter()
            # store according to type for linear vs. nonlinear comparison
            if cell_ix == 0:
                lscore_diffed[i] = lin_score
                print("Diffed")
            elif cell_ix == 1:
                lscore_rect[i] = lin_score
                print("Rectified")
            elif cell_ix == 2:
                lscore_tanhpow[i] = lin_score
                print("Tanhpow")
            elif cell_ix == 3:
                lscore_conved[i] = lin_score
                print("Convolved")
            else:
                raise Exception("Unknown response type encountered")
            print(f"Complexity calculation took {int(stop - start)} seconds; Linear model score: {lin_score}")
            print(f"2nd order model score: {sq_score}")
            start = time.perf_counter()
            avg_curve = avg_directional_curvature(m, regs, look_ahead, analyze_every)[0]
            all_crv.append(avg_curve)
            stop = time.perf_counter()
            print(f"Curvature calculation took {int(stop - start)} seconds; Avg curvature: {avg_curve}")
            start = time.perf_counter()
            nlc = calc_nlc(regs, m)[0]
            all_nlc.append(nlc)
            stop = time.perf_counter()
            print(f"NLC calculation took {int(stop - start)} seconds; NLC: {nlc}")

        print()

    all_lscore = np.hstack(all_lscore)
    all_sqscore = np.hstack(all_sqscore)
    all_crv = np.hstack(all_crv)
    all_nlc = np.hstack(all_nlc)

    # linear model scrore across conditions
    pl.figure()
    sns.kdeplot(data=lscore_diffed, cut=0, label='derivative')
    sns.kdeplot(data=lscore_rect, cut=0, label='rectified')
    sns.kdeplot(data=lscore_tanhpow, cut=0, label='$tanh^n$')
    sns.kdeplot(data=lscore_conved, cut=0, label='convolved')
    pl.legend()
    pl.xlabel("Linear model metric")
    pl.ylabel("Density")
    sns.despine()

    score_bins = np.linspace(0, 1, 100)
    fig = pl.figure()
    pl.hist(lscore_diffed, score_bins, density=True, label='derivative', histtype='step')
    pl.hist(lscore_rect, score_bins, density=True, label='rectified', histtype='step', linestyle='dashed')
    pl.hist(lscore_tanhpow, score_bins, density=True, label='$tanh^n$', histtype='step', linestyle='dashed')
    pl.hist(lscore_conved, score_bins, density=True, label='convolved', histtype='step')
    pl.legend()
    pl.xlabel("Linear model metric")
    pl.ylabel("Density")
    sns.despine()
    fig.savefig(path.join(plot_dir, "Linear_model_score_Distribution.pdf"))

    # ROC analysis
    all_linear_score = np.hstack((lscore_diffed, lscore_conved))
    all_nonlinear_score = np.hstack((lscore_rect, lscore_tanhpow))
    # test thresholds
    ttt_score = np.linspace(0, 1, 1000)

    # calculate true and false positive rates
    tpr_score, fpr_score = roc_analysis(all_linear_score, all_nonlinear_score, ttt_score)

    auc_crv = np.round(auc(fpr_score, tpr_score), 2)

    fig = pl.figure()
    pl.plot(fpr_score, tpr_score, label=f"Linear model score. AUC={auc_crv}")
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlabel("False positive rate")
    pl.ylabel("True positive rate")
    pl.xlim(0, 1.05)
    pl.ylim(0, 1.05)
    pl.legend()
    sns.despine()
    fig.savefig(path.join(plot_dir, "ROC_Analysis.pdf"))

    # plot both the true-positive rate and false-positive rate for using the linear model scrore by threshold
    fig = pl.figure()
    pl.plot(ttt_score, 1-tpr_score, label="False negative")
    pl.plot(ttt_score, fpr_score, label="False positive")
    pl.plot([0, 1], [0.05, 0.05], 'k--')
    pl.plot([0.8, 0.8], [0, 1], 'k--')
    pl.xlabel("Linear model score threshold ($R^2$)")
    pl.ylabel("Rate")
    pl.ylim(0, 1.05)
    pl.yticks([0, 0.05, 0.25, 0.5, 0.75, 1.0])
    pl.xlim(-0.05, 1.05)
    pl.xticks([0, 0.25, 0.5, 0.8, 1.0])
    pl.legend()
    sns.despine()
    fig.savefig(path.join(plot_dir, "LinearModelScore_performance.pdf"))

    # plot scatters relating 1st and 2nd order model scores
    fig = pl.figure()
    pl.plot([0, 1], [0, 1], 'k--')
    pl.scatter(all_lscore, all_sqscore, s=4, alpha=0.5)
    pl.xlabel("Linear model scores ($R^2$)")
    pl.ylabel("2nd order model scores ($R^2$)")
    sns.despine()
    fig.savefig(path.join(plot_dir, "2ndorder_vs_linear_scores.pdf"))

    # plot relationship of nonlinearity metrics and 1st order model score
    lscore_test_bins = mquantiles(all_lscore, np.linspace(0, 1, 50))
    lscore_test_bin_cents = lscore_test_bins[:-1] + np.diff(lscore_test_bins) / 2

    boot_sample = utilities.bootstrap_binned_average(all_lscore, all_crv, lscore_test_bins, 5000)
    average = np.nanmean(boot_sample, 0)
    ci_low = np.nanpercentile(boot_sample, 2.5, 0)
    ci_high = np.nanpercentile(boot_sample, 97.5, 0)
    fig = pl.figure()
    pl.plot(lscore_test_bin_cents, average, 'o')
    pl.fill_between(lscore_test_bin_cents, ci_low, ci_high, alpha=0.4)
    pl.xlabel("Linear model score")
    pl.ylabel("Average Curvature")
    sns.despine()
    fig.savefig(path.join(plot_dir, f"Avg_Crv_by_lscore.pdf"))

    boot_sample = utilities.bootstrap_binned_average(all_lscore, all_nlc, lscore_test_bins, 5000)
    average = np.nanmean(boot_sample, 0)
    ci_low = np.nanpercentile(boot_sample, 2.5, 0)
    ci_high = np.nanpercentile(boot_sample, 97.5, 0)
    fig = pl.figure()
    pl.plot(lscore_test_bin_cents, average, 'o')
    pl.fill_between(lscore_test_bin_cents, ci_low, ci_high, alpha=0.4)
    pl.xlabel("Linear model score")
    pl.ylabel("Average NLC")
    sns.despine()
    fig.savefig(path.join(plot_dir, f"Avg_NLC_by_lscore.pdf"))

    with h5py.File(path.join(plot_dir, "cnn_complx_test_data.hdf5"), 'w') as dfile:
        dfile.create_dataset("all_filters", data=np.vstack(all_filters))
        dfile.create_dataset("lscore_diffed", data=lscore_diffed)
        dfile.create_dataset("lscore_rect", data=lscore_rect)
        dfile.create_dataset("lscore_tanhpow", data=lscore_tanhpow)
        dfile.create_dataset("lscore_conved", data=lscore_conved)
        dfile.create_dataset("all_lscore", data=all_lscore)
        dfile.create_dataset("all_sqscore", data=all_sqscore)
        dfile.create_dataset("all_crv", data=all_crv)
        dfile.create_dataset("all_nlc", data=all_nlc)
