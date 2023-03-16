"""
Uses simulation to try and estimate the effective degrees of freedom
of the CNN used by MINE in the 1-predictor case on random-wave stimuli
and responses as used in cnn_sta_test.py, by estimating the covariance
between changes in true outcomes y drawn from N(0,s) and the changes
in their prediction.
See eq. (7.33; p. 233) in Hastie, Tibhirani & Friedman, 2009
"""

import numpy as np
import model
from sta_utils import gen_slow_stim, generate_response
from typing import Tuple, List
import utilities
import os
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sns


def fit_and_eval(st: np.ndarray, re: np.ndarray, evar: float, hist_steps: int,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits the model to a set of conditions and evaluates predictions
    :param st: The model input
    :param re: The response to fit
    :param evar: The variance of the errors
    :param hist_steps: The history length of the model
    :param kwargs: Optional parameters "drop_rate" and "l1_sparsity" to be used to initialize the model and "n_epochs"
        to set training epochs
    :return:
        [0]: The responses plus modulation of the current round (target)
        [1]: The model prediction of the current round
    """
    # fit model to data
    mod = np.random.randn(*re.shape) * np.sqrt(evar)
    data = utilities.Data(hist_steps, [st], re+mod)
    tset = data.training_data(0, batch_size=256)
    if "drop_rate" in kwargs:
        dr = kwargs["drop_rate"]
    else:
        dr = 0.5
    if "l1_sparsity" in kwargs:
        l1 = kwargs["l1_sparsity"]
    else:
        l1 = 1e-3
    if "n_epochs" in kwargs:
        ne = kwargs["n_epochs"]
    else:
        ne = 100
    m = model.ActivityPredictor(64, 80, dr, hist_steps, "swish")
    m.learning_rate = 1e-3
    m.l1_sparsity = l1
    m.setup()
    # the following is required to init variables at desired shape
    m(np.random.randn(1, hist_steps, 1).astype(np.float32))
    model.train_model(m, tset, ne, re.shape[1])
    prediction, target = data.predict_response(0, m)
    return target, prediction


def df(t: np.ndarray, p: np.ndarray, evar: float) -> float:
    # compute covariances and from that effective degrees of freedom
    sum_cov = 0
    for i in range(t.shape[1]):
        sum_cov += np.cov(t[:, i], p[:, i])[0, 1]
    return sum_cov / evar


def compute_edf(nframes: int, hist_steps: int, evar: float, nsims: int, fl: np.ndarray,
                fnl: np.ndarray, **kwargs) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes the effective degrees of freedom
    :param nframes: The number of frames in stimulus/response
    :param hist_steps: The history length
    :param evar: The intended error variance
    :param nsims: The number of simulations to run
    :param fl: 1st order kernel
    :param fnl: 2nd order kernel
    :param kwargs: Optional parameters "drop_rate" and "l1_sparsity" to be used to initialize the model and "n_epochs"
        to set training epochs
    :return:
        [0]: The effective degrees of freedom
        [1]: All true targets
        [2]: All predictions
    """
    time_base = np.arange(nframes) / 5
    stimulus = gen_slow_stim(time_base)

    response = generate_response(stimulus, fl, fnl)
    response -= np.mean(response)
    response /= np.std(response)
    response = response[None, :]

    all_targets = np.full((nsims, response.size-hist_steps+1), np.nan)
    all_predictions = np.full((nsims, response.size-hist_steps+1), np.nan)
    # run actual simulations
    for i in range(nsims):
        t, p = fit_and_eval(stimulus, response, evar, hist_steps, **kwargs)
        all_targets[i, :] = t
        all_predictions[i, :] = p
        if i >= 5 and i % 5 == 0:
            print(f"Edf estimate after iteration {i} = {df(all_targets[:i, :], all_predictions[:i, :], evar)}")
            print([f"{k}: {kwargs[k]}" for k in kwargs])
            print()

    # compute effective degrees of freedom
    edf = df(all_targets, all_predictions, evar)

    return edf, all_targets, all_predictions


def main(n_runs, **kwargs) -> Tuple[np.ndarray, List, List]:
    """
    Runs main program loop
    :param n_runs: The number of independent edf computations to run
    :param kwargs: Optional parameters "drop_rate" and "l1_sparsity" to be used to initialize the model and "n_epochs"
        to set training epochs
    :return:
        [0]: n_runs long vector of computed effective degrees of freedom
        [1]: n_runs element list of targets
        [2]: n_runs element list of predictions
    """
    # generate same filters, type of stimulus and response as in cnn_sta_test.py
    frame_rate = 5
    hist_seconds = 10
    h_steps = hist_seconds * frame_rate
    n_simulations = 100

    n_frames = 3000*frame_rate

    f_lin = np.zeros(h_steps)
    f_lin[:25] = 1
    f_lin[25:] = -1
    f_lin = f_lin / np.linalg.norm(f_lin)
    f_2nd = np.zeros(h_steps)
    f_2nd[20:30] = 1
    f_2nd[:10] = -1
    f_2nd[40:] = -1
    f_2nd = f_2nd / np.linalg.norm(f_2nd)

    # corrupt response with noise so that y = f(x) + e    e ~ N(0,se**2)
    error_variance = 0.25 ** 2

    all_edf = []
    all_targets = []
    all_predictions = []

    for i in range(n_runs):
        edf, t, p = compute_edf(n_frames, h_steps, error_variance, n_simulations, f_lin, f_2nd, **kwargs)
        all_edf.append(edf)
        all_targets.append(t)
        all_predictions.append(p)

    return np.hstack(all_edf), all_targets, all_predictions


if __name__ == '__main__':
    # the following will prevent tensorflow from using the GPU - as the used models have very low complexity
    # they will generally be fit faster on the CPU - comment out to run on the GPU instead
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    pdir = "mine_edf_plots"
    if not os.path.exists(pdir):
        os.makedirs(pdir)

    runs = 10
    res_dict = {"Condition": [], "N Epochs": [], "EDF": []}

    # Standard model
    mine_edf = main(runs)[0]
    for d in mine_edf:
        res_dict["Condition"].append("Standard")
        res_dict["N Epochs"].append(100)
        res_dict["EDF"].append(d)

    mine_edf = main(runs, n_epochs=1000)[0]
    for d in mine_edf:
        res_dict["Condition"].append("Standard")
        res_dict["N Epochs"].append(1000)
        res_dict["EDF"].append(d)

    # No Dropout
    mine_edf = main(runs, drop_rate=0)[0]
    for d in mine_edf:
        res_dict["Condition"].append("No Dropout")
        res_dict["N Epochs"].append(100)
        res_dict["EDF"].append(d)

    mine_edf = main(runs, drop_rate=0, n_epochs=1000)[0]
    for d in mine_edf:
        res_dict["Condition"].append("No Dropout")
        res_dict["N Epochs"].append(1000)
        res_dict["EDF"].append(d)

    # No Regularization
    mine_edf = main(runs, l1_sparsity=0)[0]
    for d in mine_edf:
        res_dict["Condition"].append("No L1")
        res_dict["N Epochs"].append(100)
        res_dict["EDF"].append(d)

    mine_edf = main(runs, l1_sparsity=0, n_epochs=1000)[0]
    for d in mine_edf:
        res_dict["Condition"].append("No L1")
        res_dict["N Epochs"].append(1000)
        res_dict["EDF"].append(d)

    # No Regularization and no drop out
    mine_edf = main(runs, l1_sparsity=0, drop_rate=0)[0]
    for d in mine_edf:
        res_dict["Condition"].append("No L1 / No Dropout")
        res_dict["N Epochs"].append(100)
        res_dict["EDF"].append(d)

    mine_edf = main(runs, l1_sparsity=0, drop_rate=0, n_epochs=1000)[0]
    for d in mine_edf:
        res_dict["Condition"].append("No L1 / No Dropout")
        res_dict["N Epochs"].append(1000)
        res_dict["EDF"].append(d)

    df_edf = pd.DataFrame(res_dict)

    fig = pl.figure()
    sns.boxplot(x="Condition", y="EDF", hue="N Epochs", data=df_edf, whis=np.inf)
    sns.despine()
    pl.yscale('log')
    fig.savefig(os.path.join(pdir, f"EDF_By_Condition.pdf"))

    f_name = os.path.join(pdir, "mine_edf.hdf5")
    df_edf.to_hdf(f_name, key="mine_cond_edf_2", mode='a')
