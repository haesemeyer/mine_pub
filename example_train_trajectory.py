
"""
Script to plot example training trajectory on the same data generation as cnn_fit_test.py
"""

import model
import numpy as np
import matplotlib.pyplot as pl
import utilities
import seaborn as sns
from scipy.signal import convolve
import matplotlib as mpl
import os


if __name__ == '__main__':
    # the following will prevent tensorflow from using the GPU - as the used models have very low complexity
    # they will generally be fit faster on the CPU - comment out to run on the GPU instead
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    mpl.rcParams['pdf.fonttype'] = 42

    plot_dir = "cnn_fit_test_plots"  # use same plot-dir as cnn_fit_test.py
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    frame_rate = 5
    n_frames = 3000
    time_base = np.arange(n_frames*5) / frame_rate
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

    tau_on = 1.4  # seconds
    tau_on *= frame_rate  # in frames
    tau_off = 2  # seconds
    tau_off *= frame_rate  # in frames
    kframes = np.arange(10 * frame_rate)  # 10 s long kernel
    kernel = 2 ** (-kframes / tau_off) * (1 - 2 ** (-kframes / tau_on))
    kernel = kernel / kernel.sum()
    ca_response = []
    for i in range(n_neurons):
        # elif resl == "MI_RS1_RS2":
        r1 = regressors[0]*2
        r2 = regressors[1]*2
        # rectify!
        r = np.log(np.exp(r1) + 1) * np.log(np.exp(r2) + 1)
        # convolve built regresssor and add to calcium response data recording appropriate label
        r = convolve(r, kernel, 'full')[:time_base.size]
        ca_response.append(r)

    # standardize calcium responses
    ca_response = np.vstack(ca_response)
    ca_response -= np.mean(ca_response, 1, keepdims=True)
    ca_response /= np.std(ca_response, 1, keepdims=True) + 1e-9
    # add noise
    ca_response += np.random.randn(*ca_response.shape) * noise_fraction

    data = utilities.Data(hist_steps, regressors, ca_response, train_frames)

    correlations_test = []  # ANN correlations on test portion of data during training trajectories
    correlations_train = []  # ANN correlations on train portion of data during training trajectories
    r = np.ndarray([])
    p = np.ndarray([])

    for cell_ix in range(ca_response.shape[0]):
        train_run = np.full(n_epochs, np.nan)
        test_run = np.full(n_epochs, np.nan)
        tset = data.training_data(cell_ix, batch_size=256)
        m = model.get_standard_model(hist_steps)
        # the following is required to init variables at desired shape
        m(np.random.randn(1, hist_steps, len(regressors)).astype(np.float32))
        # train
        for i in range(n_epochs):
            model.train_model(m, tset, 1, ca_response.shape[1])
            # evaluate
            p, r = data.predict_response(cell_ix, m)
            c_tr = np.corrcoef(p[:train_frames], r[:train_frames])[0, 1]**2
            c_ts = np.corrcoef(p[train_frames:], r[train_frames:])[0, 1]**2
            train_run[i] = c_tr
            test_run[i] = c_ts
            print(f"Epoch {i+1}  on cell {cell_ix+1} completed. Test correlation {c_ts}")
        correlations_train.append(train_run)
        correlations_test.append(test_run)

    fig = pl.figure()
    for i in range(n_neurons):
        pl.plot(np.arange(n_epochs) + 1, correlations_train[i], 'k', lw=0.25)
        pl.plot(np.arange(n_epochs) + 1, correlations_test[i], 'C1', lw=0.25)
    pl.xlabel("Epochs")
    pl.ylabel("R2")
    pl.xlim(0, n_epochs)
    pl.xticks([0, 25, 50, 75, 100])
    sns.despine()
    fig.savefig(os.path.join(plot_dir, "ANN_Train_Trajectories.pdf"))

    fig = pl.figure()
    pl.plot(time_base[train_frames+49:], r[train_frames:], 'C2', label="Real")
    pl.plot(time_base[train_frames+49:], p[train_frames:], 'C1', label="Network prediction")
    pl.xlabel("Time [s]")
    pl.ylabel("Response")
    pl.xlim(2000, 3000)
    pl.xticks([2000, 2250, 2500, 2750, 3000])
    pl.legend()
    sns.despine()
    fig.savefig(os.path.join(plot_dir, "Example_Test_Fit.pdf"))
