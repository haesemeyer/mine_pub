"""
Utility functions specific to cnn_sta_test and mine_edf to avoid circular imports
"""

import numpy as np
import utilities


def sta_nonlin(x: np.ndarray) -> np.ndarray:
    return x / (1+np.exp(-5*x))


def stc_nonlin(x: np.ndarray) -> np.ndarray:
    # constant multiplier puts output of nonlinear filter in
    # similar range as that of the linear filter
    return (0.25 - np.exp(x) / (1+np.exp(x))**2)*15


def generate_response(stim: np.ndarray, f_sta: np.ndarray, f_stc: np.ndarray) -> np.ndarray:
    if f_stc.ndim == 1:
        f_stc = f_stc[None, :]
    # have filters together with their nonlinearity act on stimulus input
    r_sta = np.convolve(stim, f_sta)[:stim.size]
    nl_sta = sta_nonlin(r_sta)
    nl_stc = np.zeros_like(nl_sta)
    for f in f_stc:
        r_stc = np.convolve(stim, f)[:stim.size]
        nl_stc += stc_nonlin(r_stc)
    # create response and return
    res = nl_sta + nl_stc
    return res


def gen_wn_stim(t: np.ndarray) -> np.ndarray:
    """
    Generates a white noise stimulus
    :param t: Timebase
    :return: t.size long vector of gaussian white noise
    """
    return np.random.randn(t.size)


def gen_slow_stim(t: np.ndarray) -> np.ndarray:
    """
    Generates a randomly but slowly varying stimulus
    :param t: Timebase
    :return: t.size long vector of random wave
    """
    stim = utilities.create_random_wave_predictor(t)
    stim -= np.mean(stim)
    stim /= np.std(stim)
    return stim
