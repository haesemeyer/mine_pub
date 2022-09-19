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


def sta_nonlin(x: np.ndarray) -> np.ndarray:
    return x / (1+np.exp(-5*x))


def stc_nonlin(x: np.ndarray) -> np.ndarray:
    return 0.25 - np.exp(x) / (1+np.exp(x))**2


def generate_response(stim: np.ndarray, f_sta: np.ndarray, f_stc: np.ndarray) -> np.ndarray:
    # have filters together with their nonlinearity act on stimulus input
    r_sta = np.convolve(stim, f_sta)[:stim.size]
    nl_sta = sta_nonlin(r_sta)
    nl_sta /= np.std(nl_sta)  # normalize so that neither sta nor stc influence dominate result
    r_stc = np.convolve(stim, f_stc)[:stim.size]
    nl_stc = stc_nonlin(r_stc)
    nl_stc /= np.std(nl_stc)  # normalize so that neither sta nor stc influence dominate result
    # create response, which would be our spike-rate
    res = nl_sta + nl_stc
    # rectify and then standardize
    res[res < 0] = 0
    return res


def main(is_white_noise: bool, whiten: bool, plot_dir: str) -> None:
    if is_white_noise:
        pfix = "WN_"
    else:
        pfix = "RW_"
    if whiten:
        pfix += "whiten_"

    frame_rate = 5
    n_frames = 3000
    time_base = np.arange(n_frames*5) / frame_rate
    hist_seconds = 10
    hist_steps = hist_seconds * frame_rate
    train_frames = (time_base.size * 2) // 3
    n_epochs = 100

    n_neurons = 100  # the number of neurons to simulate

    # create simple orthogonal filters - one for sta with a rectifying nonlinearity and
    # one for STC with a symmetric one
    sta_filter = np.zeros(hist_steps)
    sta_filter[:25] = 1
    sta_filter[25:] = -1
    sta_filter = sta_filter / np.linalg.norm(sta_filter)  # set filter to unit norm for later comparison to extracts
    stc_filter = np.zeros(hist_steps)
    stc_filter[20:30] = 1
    stc_filter[:10] = -1
    stc_filter[40:] = -1
    stc_filter = stc_filter / np.linalg.norm(stc_filter)  # set filter to unit norm for later comparison to extracts

    all_mine_sta = []  # approximation of STA extracted via MINE
    all_st_sta = []  # approximation of STA extracted via spike triggered analysis
    all_mine_stc = []  # approximation of STC filter extracted via MINE
    all_st_stc = []  # approximation of STC filter extracted via spike triggered analysis

    all_mine_sta_sims = []  # for all MINE extracted STAs the cosine similarity to the real STA
    all_st_sta_sims = []  # for all spike triggered analysis extracted STAs the cosine similarity to the real STA
    all_mine_stc_sims = []  # for all MINE extracted STC filters the cosine similarity to the real STC filter
    all_st_stc_sims = []  # for all spike triggered analysis extracted STC filters the cosine similarity to the real STC

    all_ran_sims_sta = []  # similarity between the real STA and a random vector of the same length
    all_ran_sims_stc = []  # similarity between the real STC and a random vector of the same length

    for i in range(n_neurons):
        # create stimulus
        if is_white_noise:
            stimulus = np.random.randn(time_base.size)
        else:
            stimulus = utilities.create_random_wave_predictor(time_base)
        stimulus -= np.mean(stimulus)
        stimulus /= np.std(stimulus)

        # plot stimulus examples
        if i == 0:
            fig = pl.figure()
            pl.plot(np.arange(2000) + 2000, stimulus[2000:4000], 'k')
            pl.xlabel("Timepoints")
            pl.ylabel("Stimulus [AU]")
            sns.despine()
            fig.savefig(os.path.join(plot_dir, f"{pfix}Example_Stimulus.pdf"))

        response = generate_response(stimulus, sta_filter, stc_filter)
        response_orig = response.copy()
        response -= np.mean(response)
        response /= np.std(response)
        response = response[None, :]

        # fit model to data
        data = utilities.Data(hist_steps, [stimulus], response, train_frames)
        tset = data.training_data(0, batch_size=256)
        m = model.get_standard_model(hist_steps)
        # the following is required to init variables at desired shape
        m(np.random.randn(1, hist_steps, 1).astype(np.float32))
        model.train_model(m, tset, n_epochs, response.shape[1])
        # compute the networks jacobian and hessian at the data mean
        all_inputs = []
        for inp, outp in tset:
            all_inputs.append(inp.numpy())
        x_bar = np.mean(np.vstack(all_inputs), 0, keepdims=True)
        jacobian, hessian = d2ca_dr2(m, x_bar)
        jacobian = np.reshape(jacobian, (hist_steps, 1)).T.ravel()
        jacobian = jacobian / np.linalg.norm(jacobian)
        hessian = np.reshape(hessian.numpy(), (x_bar.shape[2] * hist_steps, x_bar.shape[2] * hist_steps))
        hessian = utilities.rearrange_hessian(hessian, 1, hist_steps)
        v, ev = np.linalg.eigh(hessian)
        # get ordering starting with *largest* eigenvalue
        ordering = np.argsort(v)[::-1]

        mine_sta = jacobian.copy()

        all_mine_sta.append(mine_sta)

        # the eigenvectors with the largest two eigenvalues of the Hessian should be the STA (as it affects variance)
        # and the STC. Between them ordering is arbitrary so identify and remove the one that matches the STA
        # most closely (according to cosine similarity)
        if np.dot(ev[:, ordering[0]], sta_filter[::-1])**2 > np.dot(ev[:, ordering[1]], sta_filter[::-1])**2:
            mine_stc = ev[:, ordering[1]] * np.sign(np.dot(ev[:, ordering[1]], stc_filter[::-1]))
        else:
            mine_stc = ev[:, ordering[0]] * np.sign(np.dot(ev[:, ordering[0]], stc_filter[::-1]))

        all_mine_stc.append(mine_stc)

        # Compute cosine similarity between the Jacobian and the STA as well as between
        # the likely STC eigenvector and the true STC filter - note that while the STA
        # has a fixed direction (i.e. the direction of the Jacobian should match the STA
        # if the method works) this is not the case for the STC since by definition its
        # nonlinearity is symmetric
        # Note: The filters have to be inverted here since they are impulse responses
        # and not receptive fields
        sta_sim = np.dot(mine_sta, sta_filter[::-1])
        stc_sim = np.dot(mine_stc, stc_filter[::-1])
        r = np.random.randn(mine_sta.size)
        ran_sim = np.dot(r / np.linalg.norm(r), sta_filter[::-1])
        all_mine_sta_sims.append(sta_sim)
        all_mine_stc_sims.append(stc_sim)
        all_ran_sims_sta.append(ran_sim)
        ran_sim = np.dot(r / np.linalg.norm(r), stc_filter[::-1])
        all_ran_sims_stc.append(ran_sim)
        # compute same quantities according to spike-triggered analysis
        all_stim = []
        weights = []
        for j in range(50, response.size):
            t = stimulus[j-49:j+1]
            all_stim.append(t * response_orig[j])
            weights.append(response_orig[j])
        all_stim = np.vstack(all_stim)
        # remove ensemble average
        all_stim -= np.mean(all_stim, 0, keepdims=True)
        weights = np.hstack(weights)
        cov_all = np.cov(all_stim.T)

        if whiten:
            # ZCA whiten
            evals_ca, evecs_ca = np.linalg.eigh(cov_all)
            u = evecs_ca
            d_1_over_sqrt = np.diag(evals_ca ** (-0.5))
            w_zca = u @ d_1_over_sqrt @ u.T
            all_stim = (w_zca @ all_stim.T).T

        triggered = all_stim * weights[:, None]
        triggered = triggered[weights > 0]

        # compute spike triggered average
        st_sta = np.sum(triggered, 0) / np.sum(weights)
        st_sta /= np.linalg.norm(st_sta)
        all_st_sta.append(st_sta)
        all_st_sta_sims.append(np.dot(st_sta, sta_filter[::-1]))

        # subtract out STA and compute STC
        triggered -= (np.sum(triggered, 0) / np.sum(weights))[None, :]
        cov = (triggered.T @ triggered) / np.sum(weights)
        v, ev = np.linalg.eigh(cov)
        ordering = np.argsort(v)[::-1]
        # in this case not entirely sure if STA is necessarily contained within the covariance matrix
        # eigenvectors - so we directly pick the vector that looks most like our stc filter and hence
        # invert the comparison
        if np.dot(ev[:, ordering[0]], stc_filter[::-1])**2 < np.dot(ev[:, ordering[1]], stc_filter[::-1])**2:
            st_stc = ev[:, ordering[1]] * np.sign(np.dot(ev[:, ordering[1]], stc_filter[::-1]))
        else:
            st_stc = ev[:, ordering[0]] * np.sign(np.dot(ev[:, ordering[0]], stc_filter[::-1]))
        all_st_stc.append(st_stc)
        all_st_stc_sims.append(np.dot(st_stc, stc_filter[::-1]))

    fig = pl.figure()
    sns.kdeplot(all_st_sta_sims, label="STA", cut=0)
    sns.kdeplot(all_mine_sta_sims, label="MINE Jacobian", cut=0)
    sns.kdeplot(all_ran_sims_sta, color='k', label="Random", cut=0)
    pl.xlabel("Cosine similarity")
    pl.ylabel("Density")
    pl.legend()
    sns.despine()
    fig.savefig(os.path.join(plot_dir, f"{pfix}fSTA_Cosine_Similarities.pdf"))

    fig = pl.figure()
    sns.kdeplot(all_st_stc_sims, label="STC Eigenvector", cut=0)
    sns.kdeplot(all_mine_stc_sims, label="MINE H Eigenvector", cut=0)
    sns.kdeplot(all_ran_sims_stc, color='k', label="Random", cut=0)
    pl.xlabel("Cosine similarity")
    pl.ylabel("Density")
    pl.legend()
    sns.despine()
    fig.savefig(os.path.join(plot_dir, f"{pfix}fSTC_Cosine_Similarities.pdf"))

    filter_time = -np.arange(hist_steps)
    fig, axes = pl.subplots(ncols=2)
    axes[0].plot(filter_time, sta_filter[::-1], 'k', label="STA")
    axes[0].plot(filter_time, np.mean(all_mine_sta, 0), "C1", label="Jacobian")
    for psta in all_mine_sta:
        axes[0].plot(filter_time, psta, "C1", lw=0.5, alpha=0.25)
    axes[0].set_xlabel("Timepoint")
    axes[0].set_ylabel("Coefficient")
    axes[0].set_ylim(-0.5, 0.5)
    axes[0].legend()
    axes[1].plot(filter_time, stc_filter[::-1], 'k', label="STC")
    axes[1].plot(filter_time, np.mean(all_mine_stc, 0), "C1", label="Hessian Ev")
    for pstc in all_mine_stc:
        axes[1].plot(filter_time, pstc, "C1", lw=0.5, alpha=0.25)
    axes[1].set_xlabel("Timepoint")
    axes[1].set_ylim(-0.5, 0.5)
    axes[1].legend()
    sns.despine()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{pfix}MINE_Filters.pdf"))

    fig, axes = pl.subplots(ncols=2)
    axes[0].plot(filter_time, sta_filter[::-1], 'k', label="STA")
    axes[0].plot(filter_time, np.mean(all_st_sta, 0), "C0", label="Spike trig. avg.")
    for psta in all_st_sta:
        axes[0].plot(filter_time, psta, "C0", lw=0.5, alpha=0.25)
    axes[0].set_xlabel("Timepoint")
    axes[0].set_ylabel("Coefficient")
    axes[0].set_ylim(-0.5, 0.5)
    axes[0].legend()
    axes[1].plot(filter_time, stc_filter[::-1], 'k', label="STC")
    axes[1].plot(filter_time, np.mean(all_st_stc, 0), "C0", label="Spike trig. cov.")
    for pstc in all_st_stc:
        axes[1].plot(filter_time, pstc, "C0", lw=0.5, alpha=0.25)
    axes[1].set_xlabel("Timepoint")
    axes[1].set_ylim(-0.5, 0.5)
    axes[1].legend()
    sns.despine()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{pfix}ST_Filters.pdf"))


if __name__ == '__main__':
    # the following will prevent tensorflow from using the GPU - as the used models have very low complexity
    # they will generally be fit faster on the CPU - comment out to run on the GPU instead
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    mpl.rcParams['pdf.fonttype'] = 42

    pdir = "cnn_sta_test_plots"
    if not os.path.exists(pdir):
        os.makedirs(pdir)

    main(True, False, pdir)
    main(False, False, pdir)
    main(False, True, pdir)
