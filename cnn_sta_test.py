"""
Comparison of parameters extracted from MINE and spike triggered analysis
"""

import model
import numpy as np
import matplotlib.pyplot as pl
import utilities
import seaborn as sns
from taylorDecomp import d2ca_dr2
import matplotlib as mpl
import os
from sklearn.linear_model import Ridge, LinearRegression
from typing import Tuple
import pandas as pd
import mine_edf
from sta_utils import generate_response, gen_wn_stim, gen_slow_stim

######################
# LIST OF CHANGES
# 1) Removed whitening since switch over to LR
# 2) Removed final response rectifiation since switch over to LR
# 3) Calculate volterra kernels via LR instead of spike-triggered quantities
# 4) Removed train frames which essentially just limited data passed to ANN without any reason (no test)
# 5) Changed question of STC to: "Can an STC like vector be found within the eigenvectors with the 2 largest eigvals"
######################


def create_matrix_q(order0: float, order1: np.ndarray, order2: np.ndarray) -> np.ndarray:
    """
    Creates the combined matrix Q described in Marmarelis, 1997
    :param order0: The 0th order kernel (k0 for Volterra, f(x_bar) for MINE)
    :param order1: The 1st order kernel (k1 for Volterra, J for MINE)
    :param order2: The 2nd order kernel (k2 for Volterra, H for MINE)
    :return: (k1.size+1) x (k1.size+1) sized matrix Q
    """
    if order1.size != order2.shape[0] or order1.size != order2.shape[1]:
        raise ValueError(f"Size mismatch, {order1.size} vs. {order2.shape}")
    q = np.empty((order1.size+1, order1.size+1))
    q[0, 0] = order0
    q[0, 1:] = order1 / 2
    q[1:, 0] = order1 / 2
    q[1:, 1:] = order2
    return q


def find_best_match_from_q(matrix_q: np.ndarray, n_eig_to_test: int, real_lin: np.ndarray,
                           real_nonlin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    From matrix q among the first n eigenvectors finds the best match for the linear and nonlinear filter
    :param matrix_q: Q matrix according to Marmarelis, 1997
    :param n_eig_to_test: The number of eigenvectors in descending order of eigenvalues to examine
    :param real_lin: The real linear filter to match to the eigenvalues
    :param real_nonlin: The real nonlinear filter to match to the eigenvalues
    :return:
        [0]: The n_timepoints long eigenvector matched to the linear filter
        [1]: The n_timepoints long eigenvector matched to the non-linear filter
    """
    # NOTE: The first element of each eigenvector is a constant term and not part of the filter!
    v, ev = np.linalg.eigh(matrix_q)
    # get ordering starting with *largest* eigenvalue
    # NOTE: There are eigenvectors with large negative eigenvalues - these are suppressive directions (leading to
    # subtraction from the output) and in our case only contribute constant values (flat except for k0)
    ordering = np.argsort(v)[::-1]
    sim_lin = np.zeros(n_eig_to_test)
    sim_nlin = np.zeros_like(sim_lin)
    for i in range(n_eig_to_test):
        sim_lin[i] = np.dot(ev[1:, ordering[i]], real_lin)
        sim_nlin[i] = np.dot(ev[1:, ordering[i]], real_nonlin)
    # unsigned comparison and sign-based sign inversion below
    # are necessary since the eigenvector can be rotated by 180 degrees
    max_ix_lin = np.argmax(np.abs(sim_lin))
    max_ix_nlin = np.argmax(np.abs(sim_nlin))
    cand_lin = ev[1:, ordering[max_ix_lin]] * np.sign(sim_lin[max_ix_lin])
    cand_nlin = ev[1:, ordering[max_ix_nlin]] * np.sign(sim_nlin[max_ix_nlin])
    # we re-normalize since they are only unit-norm when including the first element
    return cand_lin/np.linalg.norm(cand_lin), cand_nlin/np.linalg.norm(cand_nlin)


def get_mine_kernels(stimulus: np.ndarray, response: np.ndarray, hist_steps: int,
                     sta_filter: np.ndarray, stc_filter: np.ndarray,
                     val_stim=None, val_res=None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Uses mine to compute approximations of the volterra filter kernels
    :param stimulus: The stimulus input
    :param response: The system response
    :param hist_steps: The number of history steps
    :param sta_filter: The actual linear filter for matching of first two evs
    :param stc_filter: The actual nonlinear filter for matching of first two evs
    :param val_stim: Optional stimulus of a validation set
    :param val_res: Optional response of a validation set
    :return:
        [0]: k1, in the same orientation as returned by the LR analysis (inverted Jacobian)
        [1]: k2, in the same orientation as returned by the LR analysis (inverted Hessian ev)
        [2]: Test correlation
    """
    # fit model to data
    data = utilities.Data(hist_steps, [stimulus], response, stimulus.size//3*2)
    tset = data.training_data(0, batch_size=256)
    m = model.get_standard_model(hist_steps)
    # the following is required to init variables at desired shape
    m(np.random.randn(1, hist_steps, 1).astype(np.float32))
    model.train_model(m, tset, 100, response.shape[1])
    # compute the networks jacobian and hessian at the data mean
    all_inputs = []
    for inp, outp in tset:
        all_inputs.append(inp.numpy())
    x_bar = np.mean(np.vstack(all_inputs), 0, keepdims=True)
    const = m(x_bar)
    jacobian, hessian = d2ca_dr2(m, x_bar)
    jacobian = np.reshape(jacobian, (hist_steps, 1)).T.ravel()
    hessian = np.reshape(hessian.numpy(), (x_bar.shape[2] * hist_steps, x_bar.shape[2] * hist_steps))
    hessian = utilities.rearrange_hessian(hessian, 1, hist_steps)
    mat_q = create_matrix_q(const, jacobian, hessian)
    # NOTE: Unlike the LR approach, the filters of interest are contained within the eigenvectors with the first
    # two eigenvalues. Nonetheless set to three to make approach the same
    mine_sta, mine_stc = find_best_match_from_q(mat_q, 3, sta_filter[::-1], stc_filter[::-1])
    # test model predictive performance
    if val_stim is None:
        p, r = data.predict_response(0, m)
        c_ts = np.corrcoef(p[stimulus.size//3*2:], r[stimulus.size//3*2:])[0, 1]
    else:
        val_data = utilities.Data(hist_steps, [val_stim], val_res)
        p, r = val_data.predict_response(0, m)
        c_ts = np.corrcoef(p, r)[0, 1]
    # return  filters time-inverted to match orientation of LR model
    return mine_sta[::-1], mine_stc[::-1], c_ts


def get_lr_kernels(stimulus: np.ndarray, response: np.ndarray, hist_steps: int, sta_filter: np.ndarray,
                   stc_filter: np.ndarray, alpha: float,
                   val_stim=None, val_res=None) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Uses linear regression to compute approximations of the volterra filter kernels
    :param stimulus: The stimulus input
    :param response: The system response
    :param hist_steps: The number of history steps
    :param sta_filter: The actual linear filter for matching of first two evs
    :param stc_filter: The actual second order filter for matching of first two evs
    :param alpha: If >0 ridge regression with this regularization will be performed instead of OLS
    :param val_stim: Optional stimulus of a validation set
    :param val_res: Optional response of a validation set
    :return:
        [0]: k1
        [1]: k2 best matching eigenvector
        [2]: Test correlation
        [3]: Effective degrees of freedom of the model
    """
    volt_y, volt_x = utilities.generate_dvs_design_matrix(response.ravel()[:stimulus.size//3*2],
                                                          stimulus.ravel()[:stimulus.size//3*2], hist_steps)
    if alpha > 0:
        df_eff = np.trace(volt_x @ np.linalg.inv(volt_x.T @ volt_x + alpha*np.identity(volt_x.shape[1])) @ volt_x.T)
    else:
        df_eff = volt_x.shape[1]
    if alpha > 0:
        lr_model = Ridge(fit_intercept=False, alpha=alpha)  # intercept is in our design matrix
    else:
        lr_model = LinearRegression(fit_intercept=False)
    lr_model.fit(volt_x, volt_y)
    vk0, vk1, vk2 = utilities.lrcoefs_to_vkernels(lr_model.coef_, hist_steps)
    mat_q = create_matrix_q(vk0, vk1, vk2)
    # The LR approach often recovers one significant filter unrelated to STA/STC in the first two positive eigenvectors
    # therefore need to search the first three
    st_sta, st_stc = find_best_match_from_q(mat_q, 3, sta_filter, stc_filter)
    if val_stim is None:
        test_y, test_x = utilities.generate_dvs_design_matrix(response.ravel()[stimulus.size//3*2:],
                                                              stimulus.ravel()[stimulus.size//3*2:], hist_steps)
        p = lr_model.predict(test_x)
        c_ts = np.corrcoef(p, test_y)[0, 1]
    else:
        test_y, test_x = utilities.generate_dvs_design_matrix(val_res.ravel(), val_stim.ravel(), hist_steps)
        p = lr_model.predict(test_x)
        c_ts = np.corrcoef(p, test_y)[0, 1]
    return st_sta, st_stc, c_ts, df_eff


def plot_kernels(df: pd.DataFrame, model_name: str, filter_type: str, true_kernel: np.ndarray) -> mpl.figure.Figure:
    time = 0 - np.arange(true_kernel.size)
    data = np.vstack(df.query(f"Model == '{model_name}' and Filter == '{filter_type}'")["Kernel"])
    figure = pl.figure()
    for sample in data:
        pl.plot(time, sample, 'C1', lw=0.25, alpha=0.25)
    pl.plot(time, np.mean(data, 0), 'C1')
    pl.plot(time, true_kernel, 'k')
    sns.despine()
    return figure


def main(is_white_noise: bool, plot_dir: str, frame_rate: int, sta_filter: np.ndarray,
         stc_filter: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # NOTE: There is a difference in time-ordering between J/H and the extracted k1/k2 from the Volterra analysis
    # While the first timepoint in k1/k2 is current time with subsequent elements going further into the past
    # current time is the *last* input to the CNN. For all plots and comparison, either the original STA/STC
    # filters or J/H-eigenvectors are therefore inverted in the ANN case but not in the Volterra case!
    n_seconds = 3000
    time_base = np.arange(n_seconds*5) / frame_rate

    hist_steps = sta_filter.size

    if is_white_noise:
        n_neurons = 10  # for white noise all conditions are essentially the same anyway
        alpha_to_test = [0]  # vanilla regression already works here anyway
    else:
        n_neurons = 100  # the number of neurons to simulate
        alpha_to_test = [0, 0.01, 0.1, 1, 10, 100, 1000, 10000]  # the regularizations to test

    # plot stimulus examples
    figure = pl.figure()
    pl.plot(np.arange(3000) + 2000, gen_wn_stim(time_base)[2000:5000], 'k', alpha=0.5, label="Gaussian white noise")
    pl.plot(np.arange(3000) + 2000, gen_slow_stim(time_base)[2000:5000], 'C3', label="Random wave")
    pl.xlabel("Timepoints")
    pl.ylabel("Stimulus [AU]")
    pl.legend()
    sns.despine()
    figure.savefig(os.path.join(plot_dir, f"Example_Stimuli.pdf"))

    result_dict = {"Model": [], "Filter": [], "Similarity": [], "Kernel": []}
    df_dict = {"Model": [], "df": [], "test correlation": []}

    for i in range(n_neurons):
        print(f"Starting iteration {i} for {'WN' if is_white_noise else 'RW'}")
        # create stimulus
        if is_white_noise:
            stimulus = gen_wn_stim(time_base)
        else:
            stimulus = gen_slow_stim(time_base)

        response = generate_response(stimulus, sta_filter, stc_filter)
        response += (np.random.randn(response.size)*np.std(response)/4)
        response -= np.mean(response)
        response /= np.std(response)
        response = response[None, :]

        mine_sta, mine_stc, test_corr = get_mine_kernels(stimulus, response, hist_steps, sta_filter, stc_filter)
        result_dict["Model"].append("MINE")
        result_dict["Kernel"].append(mine_sta)
        result_dict["Similarity"].append(np.dot(mine_sta, sta_filter))
        result_dict["Filter"].append("Linear")
        result_dict["Model"].append("MINE")
        result_dict["Kernel"].append(mine_stc)
        result_dict["Similarity"].append(np.dot(mine_stc, stc_filter))
        result_dict["Filter"].append("Non-Linear")
        df_dict["Model"].append("MINE")
        df_dict["df"].append(np.nan)
        df_dict["test correlation"].append(test_corr)

        r = np.random.randn(mine_sta.size)
        result_dict["Model"].append("Random")
        result_dict["Kernel"].append(np.full(r.size, np.nan))
        result_dict["Similarity"].append(np.dot(r / np.linalg.norm(r), sta_filter))
        result_dict["Filter"].append("Linear")
        result_dict["Model"].append("Random")
        result_dict["Kernel"].append(np.full(r.size, np.nan))
        result_dict["Similarity"].append(np.dot(r / np.linalg.norm(r), stc_filter))
        result_dict["Filter"].append("Non-Linear")

        for a in alpha_to_test:
            vk1, st_stc, test_corr, df = get_lr_kernels(stimulus, response, hist_steps, sta_filter, stc_filter, a)
            if a > 0:
                result_dict["Model"].append(f"Ridge alpha={a}")
                df_dict["Model"].append(f"Ridge alpha={a}")
            else:
                result_dict["Model"].append("OLS")
                df_dict["Model"].append(f"OLS")
            df_dict["df"].append(df)
            df_dict["test correlation"].append(test_corr)
            result_dict["Kernel"].append(vk1)
            result_dict["Similarity"].append(np.dot(vk1, sta_filter))
            result_dict["Filter"].append("Linear")
            if a > 0:
                result_dict["Model"].append(f"Ridge alpha={a}")
            else:
                result_dict["Model"].append("OLS")
            result_dict["Kernel"].append(st_stc)
            result_dict["Similarity"].append(np.dot(st_stc, stc_filter))
            result_dict["Filter"].append("Non-Linear")

    return pd.DataFrame(result_dict), pd.DataFrame(df_dict)


def data_len_test(frame_rate: int, sta_filter: np.ndarray, stc_filter: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compares between MINE and the Ridge Model with alpha=1000 (best fit) how filter quality and test correlation
    changes with changing data length
    :param frame_rate: Frame rate to create a time-base for RandomWave stimuli
    :param sta_filter: The linear filter
    :param stc_filter: The nonlinear filter
    :return: Data frame with fitting results
    """
    n_neurons = 50
    # Note: seconds_to_test includes train (2/3) and test period (1/3)
    # Note: The rather wonky numbers below are chosen because our training dataset generation drops
    # remainders that do not fit into the batch size of 256. So to not penalize MINE (by having less t-samples
    # than the LM model) we set the samples such that this drop will be minimzed
    base_seconds = 78+15
    seconds_to_test = [base_seconds, base_seconds*2, base_seconds*4, base_seconds*8, base_seconds*16, base_seconds*32,
                       base_seconds*64]
    result_dict = {"Model": [], "Filter": [], "Similarity": [], "Train frames": []}
    tcorr_dict = {"Model": [], "test correlation": [], "Train frames": []}
    hist_steps = sta_filter.size
    time_base = np.arange(seconds_to_test[-1] * 5) / frame_rate

    # to avoid that periodicity in our stimulus will result in "test data" to be in fact the same as "training data"
    # create a separate validation stimulus response pair of 1/3 the length of the longest stimulus
    val_stim = gen_slow_stim(time_base)[:time_base.size//3]
    val_res = generate_response(val_stim, sta_filter, stc_filter)
    val_res += (np.random.randn(val_res.size) * np.std(val_res) / 4)
    val_res -= np.mean(val_res)
    val_res /= np.std(val_res)
    val_res = val_res[None, :]

    for i in range(n_neurons):
        print(f"Started replicate {i+1} of {n_neurons}")
        stim_full = gen_slow_stim(time_base)
        res_full = generate_response(stim_full, sta_filter, stc_filter)
        res_full += (np.random.randn(res_full.size) * np.std(res_full) / 4)
        res_full -= np.mean(res_full)
        res_full /= np.std(res_full)
        res_full = res_full[None, :]
        for n_seconds in seconds_to_test:
            stimulus = stim_full[:n_seconds*frame_rate]
            response = res_full[:, :n_seconds*frame_rate]
            train_frames = (stimulus.size*2)//3  # for bookkeeping - our fit functions use the same logic internally
            print(f"Testing with {train_frames} frames in the training data")
            # MINE
            mine_sta, mine_stc, test_corr = get_mine_kernels(stimulus, response, hist_steps, sta_filter, stc_filter,
                                                             val_stim, val_res)
            result_dict["Model"].append("MINE")
            result_dict["Similarity"].append(np.dot(mine_sta, sta_filter))
            result_dict["Filter"].append("Linear")
            result_dict["Train frames"].append(train_frames)
            result_dict["Model"].append("MINE")
            result_dict["Similarity"].append(np.dot(mine_stc, stc_filter))
            result_dict["Filter"].append("Non-Linear")
            result_dict["Train frames"].append(train_frames)
            tcorr_dict["Model"].append("MINE")
            tcorr_dict["test correlation"].append(test_corr)
            tcorr_dict["Train frames"].append(train_frames)
            print(f"MINE test correlation = {np.round(test_corr, 2)}")
            # Ridge model with alpha=1000
            vk1, st_stc, test_corr, df = get_lr_kernels(stimulus, response, hist_steps, sta_filter, stc_filter, 1000.0,
                                                        val_stim, val_res)
            result_dict["Model"].append("Ridge alpha=1000")
            result_dict["Similarity"].append(np.dot(vk1, sta_filter))
            result_dict["Filter"].append("Linear")
            result_dict["Train frames"].append(train_frames)
            result_dict["Model"].append("Ridge alpha=1000")
            result_dict["Similarity"].append(np.dot(st_stc, stc_filter))
            result_dict["Filter"].append("Non-Linear")
            result_dict["Train frames"].append(train_frames)
            tcorr_dict["Model"].append("Ridge alpha=1000")
            tcorr_dict["test correlation"].append(test_corr)
            tcorr_dict["Train frames"].append(train_frames)
            print(f"Ridge test correlation = {np.round(test_corr, 2)}")

            vk1, st_stc, test_corr, df = get_lr_kernels(stimulus, response, hist_steps, sta_filter, stc_filter, 100.0,
                                                        val_stim, val_res)
            result_dict["Model"].append("Ridge alpha=100")
            result_dict["Similarity"].append(np.dot(vk1, sta_filter))
            result_dict["Filter"].append("Linear")
            result_dict["Train frames"].append(train_frames)
            result_dict["Model"].append("Ridge alpha=100")
            result_dict["Similarity"].append(np.dot(st_stc, stc_filter))
            result_dict["Filter"].append("Non-Linear")
            result_dict["Train frames"].append(train_frames)
            tcorr_dict["Model"].append("Ridge alpha=100")
            tcorr_dict["test correlation"].append(test_corr)
            tcorr_dict["Train frames"].append(train_frames)
            print(f"Ridge test correlation = {np.round(test_corr, 2)}")
    return pd.DataFrame(result_dict), pd.DataFrame(tcorr_dict)


if __name__ == '__main__':
    # the following will prevent tensorflow from using the GPU - as the used models have very low complexity
    # they will generally be fit faster on the CPU - comment out to run on the GPU instead
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    mpl.rcParams['pdf.fonttype'] = 42

    pdir = "cnn_sta_test_plots"
    if not os.path.exists(pdir):
        os.makedirs(pdir)

    f_rate = 5
    hist_seconds = 10
    h_steps = hist_seconds * f_rate

    # create simple orthogonal filters - one for sta with a rectifying nonlinearity and
    # one for STC with a symmetric one
    f_lin = np.zeros(h_steps)
    f_lin[:25] = 1
    f_lin[25:] = -1
    f_lin = f_lin / np.linalg.norm(f_lin)  # set filter to unit norm for later comparison to extracts
    f_2nd = np.zeros(h_steps)
    f_2nd[20:30] = 1
    f_2nd[:10] = -1
    f_2nd[40:] = -1
    f_2nd = f_2nd / np.linalg.norm(f_2nd)  # set filter to unit norm for later comparison to extracts

    df_wn, df_edf_wn = main(True, pdir, f_rate, f_lin, f_2nd)

    fig = pl.figure()
    sns.boxplot(x="Model", y="Similarity", hue="Filter", data=df_wn, whis=np.inf)
    sns.despine()
    fig.savefig(os.path.join(pdir, f"WhiteNoise_FilterSims.pdf"))

    fig = pl.figure(figsize=(15, 4.8))
    sns.boxplot(x="Model", y="test correlation", data=df_edf_wn, whis=np.inf)
    sns.despine()
    fig.savefig(os.path.join(pdir, f"WhiteNoise_TestCorrelation.pdf"))

    df_rw, df_edf = main(False, pdir, f_rate, f_lin, f_2nd)

    # add estimation of MINE effective degrees of freedom on the same stimulus/filter combination
    d_edf_mine = {"Model": [], "df": [], "test correlation": []}
    edf_mine = mine_edf.main(25)[0]
    for em in edf_mine:
        d_edf_mine["Model"].append("MINE")
        d_edf_mine["df"].append(em)
        d_edf_mine["test correlation"].append(np.nan)
    df_edf = pd.concat([df_edf, pd.DataFrame(d_edf_mine)], axis=0)

    fig = pl.figure(figsize=(20, 4.8))
    sns.boxplot(x="Model", y="Similarity", hue="Filter", data=df_rw, whis=np.inf)
    sns.despine()
    fig.savefig(os.path.join(pdir, f"RandomWave_FilterSims.pdf"))

    fig = pl.figure(figsize=(15, 4.8))
    sns.boxplot(x="Model", y="df", data=df_edf, whis=np.inf)
    sns.despine()
    fig.savefig(os.path.join(pdir, f"RandomWave_EffectiveDF.pdf"))

    fig = pl.figure(figsize=(15, 4.8))
    sns.boxplot(x="Model", y="test correlation", data=df_edf, whis=np.inf)
    sns.despine()
    fig.savefig(os.path.join(pdir, f"RandomWave_TestCorrelation.pdf"))

    fig = plot_kernels(df_rw, "MINE", "Linear", f_lin)
    fig.savefig(os.path.join(pdir, f"RW_MINE_Linear_KernelEstimate.pdf"))

    fig = plot_kernels(df_rw, "MINE", "Non-Linear", f_2nd)
    fig.savefig(os.path.join(pdir, f"RW_MINE_NonLinear_KernelEstimate.pdf"))

    fig = plot_kernels(df_rw, "Ridge alpha=1", "Linear", f_lin)
    fig.savefig(os.path.join(pdir, f"RW_RidgeAlpha1_Linear_KernelEstimate.pdf"))

    fig = plot_kernels(df_rw, "Ridge alpha=1", "Non-Linear", f_2nd)
    fig.savefig(os.path.join(pdir, f"RW_RidgeAlpha1_NonLinear_KernelEstimate.pdf"))

    fig = plot_kernels(df_rw, "Ridge alpha=100", "Linear", f_lin)
    fig.savefig(os.path.join(pdir, f"RW_RidgeAlpha100_Linear_KernelEstimate.pdf"))

    fig = plot_kernels(df_rw, "Ridge alpha=100", "Non-Linear", f_2nd)
    fig.savefig(os.path.join(pdir, f"RW_RidgeAlpha100_NonLinear_KernelEstimate.pdf"))

    fig = plot_kernels(df_rw, "Ridge alpha=1000", "Linear", f_lin)
    fig.savefig(os.path.join(pdir, f"RW_RidgeAlpha1000_Linear_KernelEstimate.pdf"))

    fig = plot_kernels(df_rw, "Ridge alpha=1000", "Non-Linear", f_2nd)
    fig.savefig(os.path.join(pdir, f"RW_RidgeAlpha1000_NonLinear_KernelEstimate.pdf"))

    # ALTERNATE FILTERS
    f_lin = np.sin(np.arange(50)/25*np.pi*2) * np.exp(-np.arange(50)*0.1)
    f_lin = f_lin / np.linalg.norm(f_lin)  # set filter to unit norm for later comparison to extracts
    f_2nd = np.sin(np.arange(50)/25*np.pi)
    f_2nd = f_2nd / np.linalg.norm(f_2nd)  # set filter to unit norm for later comparison to extracts
    f_2nd = f_2nd - f_lin*np.dot(f_2nd, f_lin)  # orthogonalize

    df_rw_alt, df_edf_alt = main(False, pdir, f_rate, f_lin, f_2nd)

    fig = pl.figure(figsize=(20, 4.8))
    sns.boxplot(x="Model", y="Similarity", hue="Filter", data=df_rw_alt, whis=np.inf)
    sns.despine()
    fig.savefig(os.path.join(pdir, f"RandomWave_FilterSims_Alternate.pdf"))

    fig = pl.figure(figsize=(15, 4.8))
    sns.boxplot(x="Model", y="test correlation", data=df_edf_alt, whis=np.inf)
    sns.despine()
    fig.savefig(os.path.join(pdir, f"RandomWave_TestCorrelation_Alternate.pdf"))

    fig = plot_kernels(df_rw_alt, "MINE", "Linear", f_lin)
    fig.savefig(os.path.join(pdir, f"RW_MINE_Linear_Alt_KernelEstimate.pdf"))

    fig = plot_kernels(df_rw_alt, "MINE", "Non-Linear", f_2nd)
    fig.savefig(os.path.join(pdir, f"RW_MINE_NonLinear_Alt_KernelEstimate.pdf"))

    fig = plot_kernels(df_rw_alt, "Ridge alpha=1", "Linear", f_lin)
    fig.savefig(os.path.join(pdir, f"RW_RidgeAlpha1_Linear_Alt_KernelEstimate.pdf"))

    fig = plot_kernels(df_rw_alt, "Ridge alpha=1", "Non-Linear", f_2nd)
    fig.savefig(os.path.join(pdir, f"RW_RidgeAlpha1_NonLinear_Alt_KernelEstimate.pdf"))

    fig = plot_kernels(df_rw_alt, "Ridge alpha=100", "Linear", f_lin)
    fig.savefig(os.path.join(pdir, f"RW_RidgeAlpha100_Linear_Alt_KernelEstimate.pdf"))

    fig = plot_kernels(df_rw_alt, "Ridge alpha=100", "Non-Linear", f_2nd)
    fig.savefig(os.path.join(pdir, f"RW_RidgeAlpha100_NonLinear_Alt_KernelEstimate.pdf"))

    fig = plot_kernels(df_rw_alt, "Ridge alpha=1000", "Linear", f_lin)
    fig.savefig(os.path.join(pdir, f"RW_RidgeAlpha1000_Linear_Alt_KernelEstimate.pdf"))

    fig = plot_kernels(df_rw_alt, "Ridge alpha=1000", "Non-Linear", f_2nd)
    fig.savefig(os.path.join(pdir, f"RW_RidgeAlpha1000_NonLinear_Alt_KernelEstimate.pdf"))

    # Test training frame dependence of test correlation and filter quality
    df_time_filter, df_time_corr = data_len_test(f_rate, f_lin, f_2nd)

    fig = pl.figure()
    sns.lineplot(x="Train frames", y="test correlation", hue="Model", data=df_time_corr, ci=68, err_style="bars")
    pl.xscale('log')
    sns.despine()
    fig.savefig(os.path.join(pdir, f"TestCorr_By_Train_Frames.pdf"))

    fig = pl.figure()
    sns.lineplot(x="Train frames", y="Similarity", hue="Model", data=df_time_filter, ci=68, err_style="bars")
    pl.xscale('log')
    sns.despine()
    fig.savefig(os.path.join(pdir, f"OverallFilterSimilarity_By_Train_Frames.pdf"))

    # save data-frames
    f_name = os.path.join(pdir, "cnn_sta_test_data.hdf5")
    df_wn.to_hdf(f_name, key="df_wn", mode='a')
    df_edf_wn.to_hdf(f_name, key="df_edf_wn", mode='a')
    df_rw.to_hdf(f_name, key="df_rw", mode='a')
    df_edf.to_hdf(f_name, key="df_edf", mode='a')
    df_rw_alt.to_hdf(f_name, key="df_rw_alt", mode='a')
    df_edf_alt.to_hdf(f_name, key="df_edf_alt", mode='a')
