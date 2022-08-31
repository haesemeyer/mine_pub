"""
Script to test nonlinearity metrics
"""

import model
import numpy as np
import matplotlib.pyplot as pl
import utilities
import seaborn as sns
import time
from taylorDecomp import avg_directional_curvature
import matplotlib as mpl
import os
from os import path
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from perf_nlc_nonlin import calc_nlc
from typing import Tuple
import h5py


def standardize(ar):
    return (ar - np.mean(ar)) / np.std(ar)


def diff_predictor(predictor: np.ndarray) -> np.ndarray:
    return np.r_[0, np.diff(predictor)]


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
        true_pos = np.sum(non_linear > th)
        true_positive_rate.append(true_pos / non_linear.size)
        false_pos = np.sum(linear > th)
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

    # granularity of nonlinearity determination
    analyze_every = frame_rate * 5  # compute every 5 seconds
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
    rectified = rect_predictor(reg)
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
    all_predictions = []
    all_true_responses = []
    all_avg_curvatures = []
    all_nlc_values = []

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
        all_predictions.append(p)
        all_true_responses.append(r)
        c_tr = np.corrcoef(p[:train_frames], r[:train_frames])[0, 1]
        correlations_trained.append(c_tr)
        c_ts = np.corrcoef(p[train_frames:], r[train_frames:])[0, 1]
        correlations_test.append(c_ts)
        stop = time.perf_counter()
        print(f"Model for neuron {cell_ix} achieved correlation {c_tr} on training data and {c_ts} on test data.")
        print(f"Training took {int(stop - start)} seconds")

        regs = data.regressor_matrix(cell_ix)
        start = time.perf_counter()
        avg_curve, all_curve = avg_directional_curvature(m, regs, look_ahead, analyze_every)
        all_avg_curvatures.append(avg_curve)
        stop = time.perf_counter()
        print(f"Curvature calculation took {int(stop - start)} seconds; Avg curvature: {avg_curve}")
        start = time.perf_counter()
        nlc = calc_nlc(regs, m)[0]
        all_nlc_values.append(nlc)
        stop = time.perf_counter()
        print(f"NLC calculation took {int(stop - start)} seconds; NLC: {nlc}")
        print()

    all_avg_curvatures = np.hstack(all_avg_curvatures)
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
    ax_curve.plot(np.arange(10)/9, all_avg_curvatures[:10], 'ko')
    ax_curve.plot(0, all_avg_curvatures[0], 'C0o')
    ax_curve.plot(5/9, all_avg_curvatures[5], 'C1o')
    ax_curve.plot(1, all_avg_curvatures[9], 'C2o')
    ax_curve.set_ylim(0, np.max(all_avg_curvatures)+0.05)
    ax_curve.set_xlabel("Nonlinear mix")
    ax_curve.set_ylabel("Curvature metric")
    ax_nlc.plot(np.arange(10)/9, all_nlc_values[:10], 'ko')
    ax_nlc.plot(0, all_nlc_values[0], 'C0o')
    ax_nlc.plot(5/9, all_nlc_values[5], 'C1o')
    ax_nlc.plot(1, all_nlc_values[9], 'C2o')
    ax_nlc.set_ylim(0, np.max(all_nlc_values)+0.05)
    ax_nlc.set_xlabel("Nonlinear mix")
    ax_nlc.set_ylabel("NLC metric")
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
    ax_curve.plot(np.arange(10)/9, all_avg_curvatures[10:20], 'ko')
    ax_curve.plot(0, all_avg_curvatures[10], 'C3o')
    ax_curve.plot(5/9, all_avg_curvatures[15], 'C4o')
    ax_curve.plot(1, all_avg_curvatures[19], 'C5o')
    ax_curve.set_ylim(0, np.max(all_avg_curvatures)+0.05)
    ax_curve.set_xlabel("Nonlinear mix")
    ax_curve.set_ylabel("Curvature metric")
    ax_nlc.plot(np.arange(10) / 9, all_nlc_values[10:20], 'ko')
    ax_nlc.plot(0, all_nlc_values[10], 'C3o')
    ax_nlc.plot(5 / 9, all_nlc_values[15], 'C4o')
    ax_nlc.plot(1, all_nlc_values[19], 'C5o')
    ax_nlc.set_ylim(0, np.max(all_nlc_values) + 0.05)
    ax_nlc.set_xlabel("Nonlinear mix")
    ax_nlc.set_ylabel("NLC metric")
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
    ax_curve.plot(np.arange(10) / 9, all_avg_curvatures[20:], 'ko')
    ax_curve.plot(0, all_avg_curvatures[20], 'C6o')
    ax_curve.plot(5 / 9, all_avg_curvatures[25], 'C7o')
    ax_curve.plot(1, all_avg_curvatures[29], 'C8o')
    ax_curve.set_ylim(0, np.max(all_avg_curvatures)+0.05)
    ax_curve.set_xlabel("Derivative mix")
    ax_curve.set_ylabel("Curvature metric")
    ax_nlc.plot(np.arange(10) / 9, all_nlc_values[20:], 'ko')
    ax_nlc.plot(0, all_nlc_values[20], 'C6o')
    ax_nlc.plot(5 / 9, all_nlc_values[25], 'C7o')
    ax_nlc.plot(1, all_nlc_values[29], 'C8o')
    ax_nlc.set_ylim(0, np.max(all_nlc_values) + 0.05)
    ax_nlc.set_xlabel("Derivative mix")
    ax_nlc.set_ylabel("NLC metric")
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

    # test stability of curvature and nlc metric across multiple random stimuli
    n_stim = 500
    curve_diffed = np.full(n_stim, np.nan)
    curve_rect = np.full(n_stim, np.nan)
    curve_tanhpow = np.full(n_stim, np.nan)
    curve_conved = np.full(n_stim, np.nan)

    nlc_diffed = np.full(n_stim, np.nan)
    nlc_rect = np.full(n_stim, np.nan)
    nlc_tanhpow = np.full(n_stim, np.nan)
    nlc_conved = np.full(n_stim, np.nan)

    all_regressors = []

    all_filters = []

    for i in range(n_stim):
        # create regressor matrix
        reg = utilities.create_random_wave_predictor(time_base)
        reg2 = standardize(utilities.create_random_wave_predictor(time_base))
        reg3 = standardize(utilities.create_random_wave_predictor(time_base))
        reg4 = standardize(utilities.create_random_wave_predictor(time_base))
        reg5 = standardize(utilities.create_random_wave_predictor(time_base))
        reg -= np.mean(reg)
        reg /= np.std(reg)
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
        s = np.random.uniform(1.0, 5.0)
        o = np.random.uniform(1.0, 5.0)
        rectified = rect_predictor(reg, s, o)
        rectified /= np.std(rectified)
        e = np.random.randint(1, 5)
        tanhsq = tanhsq_predictor(reg, e)
        tanhsq /= np.std(tanhsq)
        diffed = diff_predictor(reg)
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
            avg_curve = avg_directional_curvature(m, regs, look_ahead, analyze_every)[0]
            stop = time.perf_counter()
            if cell_ix == 0:
                curve_diffed[i] = avg_curve
            elif cell_ix == 1:
                curve_rect[i] = avg_curve
            elif cell_ix == 2:
                curve_tanhpow[i] = avg_curve
            elif cell_ix == 3:
                curve_conved[i] = avg_curve
            else:
                raise Exception("Unknown response type encountered")
            print(f"Curvature calculation took {int(stop - start)} seconds; Avg curvature: {avg_curve}")
            start = time.perf_counter()
            nlc = calc_nlc(regs, m)[0]
            stop = time.perf_counter()
            if cell_ix == 0:
                nlc_diffed[i] = nlc
            elif cell_ix == 1:
                nlc_rect[i] = nlc
            elif cell_ix == 2:
                nlc_tanhpow[i] = nlc
            elif cell_ix == 3:
                nlc_conved[i] = nlc
            else:
                raise Exception("Unknown response type encountered")
            print(f"NLC calculation took {int(stop - start)} seconds; NLC: {nlc}")
        print()

    # curvature across conditions
    fig = pl.figure()
    sns.kdeplot(data=curve_diffed, cut=0, label='derivative')
    sns.kdeplot(data=curve_rect, cut=0, label='rectified')
    sns.kdeplot(data=curve_tanhpow, cut=0, label='$tanh^n$')
    sns.kdeplot(data=curve_conved, cut=0, label='convolved')
    pl.legend()
    pl.xlabel("Curvature metric")
    pl.ylabel("Density")
    sns.despine()
    fig.savefig(path.join(plot_dir, "CurvatureDistribution.pdf"))

    # NLC across conditions
    fig = pl.figure()
    sns.kdeplot(data=nlc_diffed, cut=0, label='derivative')
    sns.kdeplot(data=nlc_rect, cut=0, label='rectified')
    sns.kdeplot(data=nlc_tanhpow, cut=0, label='$tanh^n$')
    sns.kdeplot(data=nlc_conved, cut=0, label='convolved')
    pl.legend()
    pl.xlabel("NLC metric")
    pl.ylabel("Density")
    sns.despine()
    fig.savefig(path.join(plot_dir, "NLCDistribution.pdf"))

    # ROC analysis
    all_linear_crv = np.hstack((curve_diffed, curve_conved))
    all_nonlinear_crv = np.hstack((curve_rect, curve_tanhpow))
    # test thresholds
    ttt_crv = np.linspace(0, np.max(np.hstack((all_linear_crv, all_nonlinear_crv))), 1000)

    all_linear_nlc = np.hstack((nlc_diffed, nlc_conved))
    all_nonlinear_nlc = np.hstack((nlc_rect, nlc_tanhpow))
    ttt_nlc = np.linspace(0, np.max(np.hstack((all_linear_nlc, all_nonlinear_nlc))), 1000)

    # calculate true and false positive rates
    tpr_crv, fpr_crv = roc_analysis(all_linear_crv, all_nonlinear_crv, ttt_crv)
    tpr_nlc, fpr_nlc = roc_analysis(all_linear_nlc, all_nonlinear_nlc, ttt_nlc)

    auc_crv = np.round(auc(fpr_crv, tpr_crv), 2)
    auc_nlc = np.round(auc(fpr_nlc, tpr_nlc), 2)

    # create logistic regression model combining curvature and NLC metric
    all_nlc = np.hstack((all_linear_nlc, all_nonlinear_nlc))
    nlc_mean = np.mean(all_nlc)
    nlc_std = np.std(all_nlc)
    all_nlc -= nlc_mean
    all_nlc /= nlc_std
    all_crv = np.hstack((all_linear_crv, all_nonlinear_crv))
    crv_mean = np.mean(all_crv)
    crv_std = np.std(all_crv)
    all_crv -= crv_mean
    all_crv /= crv_std
    X = np.hstack((all_crv[:, None], all_nlc[:, None]))
    ix = np.arange(X.shape[0])
    np.random.shuffle(ix)
    ix_train = ix[:ix.size//2]
    ix_test = ix[ix.size//2:]
    out_labels = np.hstack((np.zeros(all_linear_nlc.size, dtype=bool), np.ones(all_nonlinear_nlc.size, dtype=bool)))
    test_true = out_labels[ix_test]
    test_false = np.logical_not(out_labels[ix_test])
    lrm = LogisticRegression()
    lrm.fit(X[ix_train], out_labels[ix_train])
    class_prob = lrm.predict_proba(X[ix_test])[:, 1]
    ttt_lrm = np.linspace(0, 1, 1000)
    tpr_lrm, fpr_lrm = roc_analysis(class_prob[test_false], class_prob[test_true], ttt_lrm)
    auc_lrm = np.round(auc(fpr_lrm, tpr_lrm), 2)

    print("Logistic regression model scalings and parameters:")
    print(f"NLC Mean: {nlc_mean}")
    print(f"NLC Standard Deviation: {nlc_std}")
    print(f"Curvature Mean: {crv_mean}")
    print(f"Curvature Standard Deviation: {crv_std}")
    print(f"LRM Curvature weight: {lrm.coef_[0, 0]}")
    print(f"LRM NLC weight: {lrm.coef_[0, 1]}")
    print(f"LRM intercept: {lrm.intercept_}")

    fig = pl.figure()
    pl.plot(fpr_crv, tpr_crv, label=f"Curvature. AUC={auc_crv}")
    pl.plot(fpr_nlc, tpr_nlc, label=f"NLC. AUC={auc_nlc}")
    pl.plot(fpr_lrm, tpr_lrm, label=f"LRM. AUC={auc_lrm}")
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlabel("False positive rate")
    pl.ylabel("True positive rate")
    pl.xlim(0, 1)
    pl.ylim(0, 1)
    pl.legend()
    sns.despine()
    fig.savefig(path.join(plot_dir, "ROC_Analysis.pdf"))

    # plot both the true-positive rate and false-positive rate for our classifier model by threshold
    fig = pl.figure()
    pl.plot(ttt_lrm, tpr_lrm, label="True positive")
    pl.plot(ttt_lrm, fpr_lrm, label="False positive")
    pl.plot([0, 1], [0.01, 0.01], 'k--')
    acceptable = np.where(fpr_lrm < 0.01)[0][0]
    pl.plot([ttt_lrm[acceptable], ttt_lrm[acceptable]], [0, 1], 'k--')
    pl.xlabel("Threshold p(nonlinear)")
    pl.ylabel("Rate")
    pl.ylim(0, 1)
    pl.yticks([0, 0.25, 0.5, 0.75, 1.0])
    pl.xlim(0, 1)
    pl.xticks([0, 0.25, 0.5, 0.75, 1.0])
    pl.legend()
    sns.despine()
    fig.savefig(path.join(plot_dir, "NL_LRM_performance.pdf"))

    with h5py.File(path.join(plot_dir, "cnn_nonlin_test_data.hdf5"), 'w') as dfile:
        dfile.create_dataset("nlc_mean", data=nlc_mean)
        dfile.create_dataset("nlc_std", data=nlc_std)
        dfile.create_dataset("crv_mean", data=crv_mean)
        dfile.create_dataset("crv_std", data=crv_std)
        dfile.create_dataset("lrm_coef", data=lrm.coef_)
        dfile.create_dataset("lrm_intercept", data=lrm.intercept_)
        dfile.create_dataset("all_filters", data=np.vstack(all_filters))
        dfile.create_dataset("curve_diffed", data=curve_diffed)
        dfile.create_dataset("curve_rect", data=curve_rect)
        dfile.create_dataset("curve_tanhpow", data=curve_tanhpow)
        dfile.create_dataset("curve_conved", data=curve_conved)
        dfile.create_dataset("nlc_diffed", data=nlc_diffed)
        dfile.create_dataset("nlc_rect", data=nlc_rect)
        dfile.create_dataset("nlc_tanhpow", data=nlc_tanhpow)
        dfile.create_dataset("nlc_conved", data=nlc_conved)
