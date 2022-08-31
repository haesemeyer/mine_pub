
"""
Main test script
"""

import model
import numpy as np
import matplotlib.pyplot as pl
import utilities
import seaborn as sns
from scipy.signal import convolve
import time
from pandas import DataFrame, concat
from taylorDecomp import taylor_decompose
import matplotlib as mpl
import os
import statsmodels.formula.api as smf


if __name__ == '__main__':
    # the following will prevent tensorflow from using the GPU - as the used models have very low complexity
    # they will generally be fit faster on the CPU - comment out to run on the GPU instead
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    mpl.rcParams['pdf.fonttype'] = 42

    plot_dir = "cnn_fit_test_plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    frame_rate = 5
    n_frames = 3000
    time_base = np.arange(n_frames*5) / frame_rate
    assert time_base.size % 3 == 0
    trial_time = time_base[:time_base.size//3]
    hist_seconds = 10
    hist_steps = hist_seconds * frame_rate
    hist_frames = hist_seconds * frame_rate
    train_frames = (time_base.size * 2) // 3
    n_epochs = 100

    # all signals are standardized to unit variance/sd - noise_fraction controls the standard deviation of the noise
    # SNR = 1/noise_fraction
    noise_fraction = 0.25

    n_neurons = 20  # the number of neurons to simulate for each predictor combination

    # granularity of taylor decomposition
    analyze_every = frame_rate * 5  # compute every 5 seconds
    look_ahead = frame_rate * 5  # use 5s look-ahead to compute input data direction

    # regressors and responses to form:
    # Noise: No relationship to inputs, gaussian noise
    # Sens_#: Sensory like, smooth regressor
    # Mot_#: Motor like, poisson regressor
    # MI_#_#: Multiplicative interaction regressor of named sensory and motor regressors
    # A_#: Absolute version of indicated regressor
    # R_#: Rectified version of indicated regressor
    # D_#: Temporal derivative of indicated regressor
    reg_labels = ["Sens_1", "Sens_2", "Mot_1", "Mot_2"]
    response_labels = reg_labels + ["Noise", "MI_S1_S2", "MI_RS1_RS2", "MI_RS1_M1", "A_S1", "T_S2", "D_S1"]

    # To keep results consistent for testing purposes - alternatively and to speed things up
    # we could save the regressors and calcium responses instead of regenerating both every time
    np.random.seed(777)

    # build regressors - create 4: 2 akin to slowly varying stimulus regressors that are active in parts
    # of each trial and 2 like behavioral regressors (stochastic response with given probability)
    regressors = []
    for i in range(2):
        reg = utilities.create_random_wave_predictor(time_base)
        reg /= np.std(reg)
        regressors.append(reg)
    reg = (np.random.rand(time_base.size) < 0.01).astype(np.float32)
    reg /= np.std(reg)
    regressors.append(reg)
    reg = (np.random.rand(time_base.size) < 0.01) * np.random.randn(time_base.size)
    reg /= np.std(reg)
    regressors.append(reg)

    fig, axes = pl.subplots(nrows=len(regressors))
    for i, reg in enumerate(regressors):
        axes[i].plot(time_base, reg, lw=1)
        axes[i].plot([time_base[-1]//3, time_base[-1]//3], [np.min(reg), np.max(reg)], "k--", lw=0.75)
        axes[i].plot([2*time_base[-1]//3, 2*time_base[-1]//3], [np.min(reg), np.max(reg)], "k--", lw=0.75)
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    sns.despine()
    fig.savefig(os.path.join(plot_dir, "Predictors.pdf"))

    # create calcium responses from regressors according to list above - generate 10 each, these will later differ
    # since noise will be added to calcium data
    tau_on = 1.4  # seconds
    tau_on *= frame_rate  # in frames
    tau_off = 2  # seconds
    tau_off *= frame_rate  # in frames
    kframes = np.arange(10 * frame_rate)  # 10 s long kernel
    kernel = 2 ** (-kframes / tau_off) * (1 - 2 ** (-kframes / tau_on))
    kernel = kernel / kernel.sum()
    ca_response = []
    cluster_membership = []  # will contain the cluster labels given above, simply go through in order
    r = None
    for resl in response_labels:
        for i in range(n_neurons):
            if resl == "Sens_1":
                r = regressors[0].copy()
            elif resl == "Sens_2":
                r = regressors[1].copy()
            elif resl == "Mot_1":
                r = regressors[2].copy()
            elif resl == "Mot_2":
                r = regressors[3].copy()
            elif resl == "MI_S1_S2":
                r1 = regressors[0]
                r2 = regressors[1]
                r = r1 * r2
            elif resl == "MI_RS1_RS2":
                r1 = regressors[0]*2
                r2 = regressors[1]*2
                # rectify!
                r = np.log(np.exp(r1) + 1) * np.log(np.exp(r2) + 1)
            elif resl == "MI_RS1_M1":
                r1 = regressors[0]*2
                r3 = regressors[2]*2
                r = np.log(np.exp(r1) + 1) * r3
            elif resl == "A_S1":
                r = regressors[0].copy()
                r = np.abs(r)
            elif resl == "T_S2":
                r = regressors[1].copy()
                r = (r > 1.0).astype(float)
            elif resl == "D_S1":
                r = np.diff(regressors[0])
            elif resl == "Noise":
                r = np.random.randn(time_base.size) * 0.25
            else:
                raise ValueError("Invalid response label")
            # convolve built regresssor and add to calcium response data recording appropriate label
            r = convolve(r, kernel, 'full')[:time_base.size]
            ca_response.append(r)
            cluster_membership.append(resl)

    # standardize calcium responses
    ca_response = np.vstack(ca_response)
    ca_response -= np.mean(ca_response, 1, keepdims=True)
    ca_response /= np.std(ca_response, 1, keepdims=True) + 1e-9
    # add noise
    ca_response += np.random.randn(*ca_response.shape) * noise_fraction

    fig = pl.figure()
    sns.heatmap(ca_response, rasterized=True)
    sns.despine()
    pl.title("Ca responses")
    fig.savefig(os.path.join(plot_dir, "Response_Heatmap.pdf"), dpi=450)

    ncol = int(np.sqrt(ca_response.shape[0]//n_neurons)) + 1
    nrow = int(np.sqrt(ca_response.shape[0]//n_neurons))
    fig, axes = pl.subplots(nrows=nrow, ncols=ncol)
    axes = axes.ravel()
    for i, res in enumerate(ca_response):
        if i % n_neurons == 0:
            axes[i//n_neurons].plot(time_base, res, lw=1)
            axes[i//n_neurons].set_title(response_labels[i//n_neurons])
            axes[i//n_neurons].set_xticks([])
            axes[i//n_neurons].set_yticks([])
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "Example_Responses.pdf"))

    data = utilities.Data(hist_steps, regressors, ca_response, train_frames)

    # compute and plot mutual information metrics between predictors and the response types
    # both for raw and convolved predictors
    mi_data_raw = np.full((data.ca_responses.shape[0], len(data.regressors)), np.nan)
    mi_data_conv = mi_data_raw.copy()
    for i, ca in enumerate(data.ca_responses):
        for j, r in enumerate(data.regressors):
            cr = convolve(r.ravel(), kernel, 'full')[:time_base.size]
            mi_data_conv[i, j] = utilities.mutual_information(ca.ravel(), cr, 25)
            mi_data_raw[i, j] = utilities.mutual_information(ca.ravel(), r.ravel(), 25)
    df_mi_raw = DataFrame(mi_data_raw, columns=reg_labels)
    df_mi_conv = DataFrame(mi_data_conv, columns=reg_labels)
    fig = pl.figure()
    sns.heatmap(df_mi_raw, yticklabels=n_neurons, cbar_kws={"label": "Mutual information [bits]"})
    pl.xlabel("Predictor")
    pl.ylabel("Response type")
    fig.savefig(os.path.join(plot_dir, "MutualInfo_Raw.pdf"))

    fig = pl.figure()
    sns.heatmap(df_mi_conv, yticklabels=n_neurons, cbar_kws={"label": "Mutual information [bits]"})
    pl.xlabel("Convolved Predictor")
    pl.ylabel("Response type")
    fig.savefig(os.path.join(plot_dir, "MutualInfo_Convolved.pdf"))

    correlations_naive = []  # ANN correlations of untrained model
    correlations_trained = []  # ANN correlations on training portion of data after training
    correlations_test = []  # ANN correlations on test portion of data after training
    lr_simple = []  # Correlations on test portion of data for simple linear regression model (no conv, terms only)
    lr_conv_expanded = []  # ... for lr model with convolved regressors and first-order interactions
    all_predictions = []
    all_true_responses = []
    model_dict = {}  # dictionariy indexed by cluster-membership. Has tuple (model, cell_ix) as content
    # Formula definitions for linear regression models
    var_names = [f"x{i}" for i in range(len(regressors))]  # input column names for raw predictors
    var_names_conv = [f"cx{i}" for i in range(len(regressors))]  # input column names for convolved predictors
    formula_simple = "y ~ " + "+".join(var_names)
    formula_expanded = "y ~ (" + "+".join(var_names_conv) + ")**2"
    for cell_ix in range(ca_response.shape[0]):
        start = time.perf_counter()
        tset = data.training_data(cell_ix, batch_size=256)
        m = model.get_standard_model(hist_steps)
        # the following is required to init variables at desired shape
        m(np.random.randn(1, hist_steps, len(regressors)).astype(np.float32))
        p, r = data.predict_response(cell_ix, m)
        correlations_naive.append(np.corrcoef(p, r)[0, 1]**2)
        # train
        model.train_model(m, tset, n_epochs, ca_response.shape[1])
        # evaluate
        p, r = data.predict_response(cell_ix, m)
        all_predictions.append(p)
        all_true_responses.append(r)
        c_tr = np.corrcoef(p[:train_frames], r[:train_frames])[0, 1]
        correlations_trained.append(c_tr**2)
        c_ts = np.corrcoef(p[train_frames:], r[train_frames:])[0, 1]
        correlations_test.append(c_ts**2)
        stop = time.perf_counter()
        print(f"Model for neuron {cell_ix} achieved correlation {c_tr} on training data and {c_ts} on test data.")
        print(f"Training took {int(stop - start)} seconds")
        # keep each well-fit model around for later analysis
        if c_ts >= 0.6:
            if cluster_membership[cell_ix] not in model_dict:
                model_dict[cluster_membership[cell_ix]] = []
            model_dict[cluster_membership[cell_ix]].append((m, cell_ix))
        # fit and evaluate OLS models
        start = time.perf_counter()
        regs = data.regressor_matrix(cell_ix)
        x_simple = DataFrame(regs, columns=var_names)
        conv_regs = regs.copy()
        for i in range(conv_regs.shape[1]):
            conv_regs[:, i] = convolve(conv_regs[:, i], kernel, 'full')[:time_base.size]
        x_expanded = DataFrame(conv_regs, columns=var_names_conv)
        y = DataFrame(data=ca_response[cell_ix], columns=['y'])
        df_simple = concat((y, x_simple), axis=1)
        df_expanded = concat((y, x_expanded), axis=1)
        lm_simple = smf.ols(formula=formula_simple, data=df_simple[:train_frames]).fit()
        lm_expanded = smf.ols(formula=formula_expanded, data=df_expanded[:train_frames]).fit()
        pred_simple = lm_simple.predict(df_simple[train_frames:])
        pred_expanded = lm_expanded.predict(df_expanded[train_frames:])
        c_simp = np.corrcoef(pred_simple, y['y'].iloc[train_frames:].values)[0, 1]
        c_exp = np.corrcoef(pred_expanded, y['y'].iloc[train_frames:].values)[0, 1]
        lr_simple.append(c_simp**2)
        lr_conv_expanded.append(c_exp**2)
        stop = time.perf_counter()
        print(f"Simple LR for neuron {cell_ix} achieved correlation {c_simp} expanded achieved {c_exp} on test data.")
        print(f"LR fits took {int(stop - start)} seconds")
        print()

    correlations_naive = np.hstack(correlations_naive)
    correlations_trained = np.hstack(correlations_trained)
    correlations_test = np.hstack(correlations_test)
    lr_simple = np.hstack(lr_simple)
    lr_conv_expanded = np.hstack(lr_conv_expanded)
    all_predictions = np.vstack(all_predictions)
    all_true_responses = np.vstack(all_true_responses)

    # build pandas dataframe of all correlation and cluster information
    info_dict = {"Cluster": [], "$R^2$": [], "Network fit": []}
    for i, cl in enumerate(cluster_membership):
        info_dict["Cluster"].append(cl)
        info_dict["Cluster"].append(cl)
        info_dict["Cluster"].append(cl)
        info_dict["$R^2$"].append(correlations_trained[i])
        info_dict["Network fit"].append("Trained")
        info_dict["$R^2$"].append(correlations_test[i])
        info_dict["Network fit"].append("Test")
        info_dict["$R^2$"].append(correlations_naive[i])
        info_dict["Network fit"].append("Naive")
    info_dframe_ann = DataFrame(info_dict)

    fig = pl.figure()
    sns.pointplot(x="$R^2$", y="Cluster", hue="Network fit", data=info_dframe_ann,
                  hue_order=["Naive", "Trained", "Test"], ci="sd", markers='.', join=False, dodge=0.25, errwidth=1)
    sns.despine()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "R2_by_type_and_training_state.pdf"))

    fig, (ax1, ax2) = pl.subplots(ncols=2)
    ax1.scatter(lr_simple, correlations_test)
    ax1.plot([-0.5, 1], [-0.5, 1], 'k--')
    ax1.set_xlabel("Linear model test $R^2$")
    ax1.set_ylabel("ANN model test $R^2$")
    ax1.set_title("Simple linear model")
    ax2.scatter(lr_conv_expanded, correlations_test)
    ax2.plot([-0.5, 1], [-0.5, 1], 'k--')
    ax2.set_xlabel("Linear model test $R^2$")
    ax2.set_ylabel("ANN model test $R^2$")
    ax2.set_title("With interactions and pre-convolution")
    sns.despine()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "R2_ANN_vs_LM.pdf"))

    # build pandas dataframe to compare LR and ANN across clusters
    info_dict = {"Cluster": [], "Test $R^2$": [], "Model Type": []}
    for i, cl in enumerate(cluster_membership):
        info_dict["Cluster"].append(cl)
        info_dict["Cluster"].append(cl)
        info_dict["Cluster"].append(cl)
        info_dict["Test $R^2$"].append(lr_simple[i])
        info_dict["Model Type"].append("LM Simple")
        info_dict["Test $R^2$"].append(correlations_test[i])
        info_dict["Model Type"].append("ANN")
        info_dict["Test $R^2$"].append(lr_conv_expanded[i])
        info_dict["Model Type"].append("LM Expanded")
    info_dframe_methods = DataFrame(info_dict)

    fig = pl.figure()
    sns.pointplot(x="Test $R^2$", y="Cluster", hue="Model Type", data=info_dframe_methods,
                  hue_order=["LM Simple", "LM Expanded", "ANN"], join=False, palette='Set2', ci='sd', markers='.',
                  dodge=0.25, errwidth=1)
    sns.despine()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "R2_by_type_and_model.pdf"))

    class_labels_orig = ["S1", "S1*S2", "S1*M1", "S1*SM2", "S2", "S2*M1", "S2*M2", "M1", "M1*M2", "M2"]
    class_colors_orig = ["C0", "C3", "C3", "C3", "C0", "C3", "C3", "C0", "C3", "C0"]

    MI_RS1_RS2_full = None
    MI_RS1_RS2_ann = None
    MI_RS1_RS2_rem_interact = None
    MI_RS1_RS2_rem_s1 = None

    for reslab in response_labels:
        if reslab not in model_dict:
            continue
        cmat_all = []
        for num, md in enumerate(model_dict[reslab]):
            m = md[0]
            regs = data.regressor_matrix(md[1])
            # predict five seconds ahead, every five seconds
            start = time.perf_counter()
            real, pred, by_reg = taylor_decompose(m, regs, analyze_every, look_ahead, True)
            stop = time.perf_counter()
            print(f"Taylor decomposition took {int(stop - start)} seconds")
            cmat = np.full((len(regressors), len(regressors)), np.nan)
            for i in range(len(regressors)):
                for j in range(len(regressors)):
                    if i == j:
                        remainder = pred - by_reg[:, i, j]
                        if reslab == "MI_RS1_RS2" and num == 0 and i == 0:
                            MI_RS1_RS2_rem_s1 = remainder.copy()
                    else:
                        remainder = pred - by_reg[:, i, j] - by_reg[:, j, i]
                        if reslab == "MI_RS1_RS2" and num == 0 and i == 0 and j == 1:
                            MI_RS1_RS2_full = pred.copy()
                            MI_RS1_RS2_ann = real.copy()
                            MI_RS1_RS2_rem_interact = remainder.copy()
                    bs = utilities.bootstrap_fractional_r2loss(real, pred, remainder, 5000)
                    bs_m = np.mean(bs)
                    if bs_m < 0 and np.percentile(bs, 99) < 0:
                        cmat[i, j] = bs_m
                    elif bs_m > 0 and np.percentile(bs, 1) > 0:
                        cmat[i, j] = bs_m
                    else:
                        cmat[i, j] = 0
            if reslab == "MI_RS1_RS2" and num == 0:
                # plot example correspondences
                r2_full = np.corrcoef(MI_RS1_RS2_full, MI_RS1_RS2_ann)[0, 1]**2
                r2_s1 = np.corrcoef(MI_RS1_RS2_rem_s1, MI_RS1_RS2_ann)[0, 1]**2
                metric_s1 = (r2_full - r2_s1) / r2_full
                r2_inter = np.corrcoef(MI_RS1_RS2_rem_interact, MI_RS1_RS2_ann)[0, 1]**2
                metric_inter = (r2_full - r2_inter) / r2_full
                fig, axes = pl.subplots(ncols=3)
                axes[0].scatter(MI_RS1_RS2_full, MI_RS1_RS2_ann, color='k', s=3)
                axes[0].set_xlabel("Taylor expansion f(dP)")
                axes[0].set_ylabel("dANN output")
                axes[0].set_title(f"$R^2 = {np.round(r2_full, 2)}$")
                axes[0].set_xlim(-5, 5)
                axes[0].set_xticks([-5, -2.5, 0, 2.5, 5])
                axes[1].scatter(MI_RS1_RS2_rem_s1, MI_RS1_RS2_ann, color='k', s=3)
                axes[1].set_xlabel("f(dP) - f(dS_1)")
                axes[1].set_title(f"$R^2 = {np.round(r2_s1, 2)}$, Metric={np.round(metric_s1, 2)}")
                axes[1].set_xlim(-5, 5)
                axes[1].set_xticks([-5, -2.5, 0, 2.5, 5])
                axes[2].scatter(MI_RS1_RS2_rem_interact, MI_RS1_RS2_ann, color='k', s=3)
                axes[2].set_xlabel("f(dP) - f(dS_1, dS_2)")
                axes[2].set_title(f"$R^2 = {np.round(r2_inter, 2)}$, Metric={np.round(metric_inter, 2)}")
                axes[2].set_xlim(-5, 5)
                axes[2].set_xticks([-5, -2.5, 0, 2.5, 5])
                sns.despine()
                fig.tight_layout()
                fig.savefig(os.path.join(plot_dir, "MI_R1_R2_TaylorExample.pdf"))

            cmat_all.append(cmat)
        if len(cmat_all) > 1:
            cmat_avg = np.mean(cmat_all, 0)
        else:
            cmat_avg = cmat_all[0]
        # cmat_all[cmat_all == 0] = np.nan
        # create barplot from individual elements in cmat, sorted by value
        class_values = cmat_avg[np.triu_indices(cmat_avg.shape[0])]
        sort_ix = np.argsort(class_values)[::-1]
        class_labels = [class_labels_orig[i] for i in sort_ix]
        cm_dict = {"Response class": [], "Taylor metric": []}
        for cm in cmat_all:
            class_values = cm[np.triu_indices(cm.shape[0])]
            for (cv, cl) in zip(class_values[sort_ix], class_labels):
                cm_dict["Response class"].append(cl)
                cm_dict["Taylor metric"].append(cv)
        cm_df = DataFrame(cm_dict)
        fig = pl.figure()
        sns.barplot(x="Response class", y="Taylor metric", data=cm_df, color='C0', ci=95)
        pl.plot([0, len(class_labels)-1], [5e-3, 5e-3], 'k--')
        pl.ylim(1e-9, 1.25)
        pl.yscale('log')
        sns.despine()
        pl.title(reslab)
        fig.savefig(os.path.join(plot_dir, f"Predictor_Importance_{reslab}_log.pdf"))
