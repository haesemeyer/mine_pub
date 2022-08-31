# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from skimage.util import view_as_windows
from model import ActivityPredictor
from typing import Tuple


# Used in combination with "cnn_fit_test.py" and its derivatives


def calc_nlc(predictors: np.ndarray, mdl: ActivityPredictor, train_frames=0) -> Tuple[float, float]:
    # Convert to numpy array if list
    if not isinstance(predictors, np.ndarray):
        predictors = np.asarray(predictors)

    # Determine shape of input array ('predictors')
    if predictors.ndim == 1:
        predictors = predictors[:, None]
    assert predictors.shape[0] != predictors.shape[
        1], 'Predictors dimensions are equal, indicating the size is either too small or big.'
    if predictors.shape[1] > predictors.shape[0]:
        predictors = np.transpose(predictors)
    win_size = (mdl.input_length, predictors.shape[1])

    # Separate the predictor matrix into windows
    if mdl.input_length > 1:
        win_reg = tf.convert_to_tensor(view_as_windows(predictors, win_size, 1).squeeze(), dtype='float32')
    else:
        win_reg = predictors
    # Adjust shape
    if win_reg.ndim == 2:
        win_reg = win_reg[:, :, None]
    # Average 'win_reg'
    avg_reg = tf.reduce_mean(win_reg, axis=0)  # x_bar
    # Calculate covariance matrix
    n = win_reg.shape[0]
    x = tf.reshape(win_reg - avg_reg[None, :, :], [n, -1])
    covx = (1 / n) * tf.matmul(tf.transpose(x), x)

    # For a given model calculate average response (ex., 'Sens_1')
    f_bar = tf.reduce_mean(mdl.get_output(win_reg))

    # Calculate Tr(Cov_f)
    trace_covf = tf.Variable([0], dtype='float32')
    for wr in win_reg[train_frames:]:
        trace_covf.assign_add((mdl.get_output(wr[None, :]) - f_bar) ** 2)  # second pass, avg_reg [=] (50,5)
    nlc_denominator = trace_covf / tf.cast(win_reg.shape[0] - train_frames, tf.float32)

    # Alternative way to calculate Tr(Cov_f)
    # p, r = data.predict_response(cell_ix, m)  ## second pass
    # alt_trace_covf = tf.norm(tf.convert_to_tensor(p[train_frames:]) - f_bar, ord=2)**2

    # Forward propagate randomly selected windows through network
    btch = tf.constant(250)  # batches
    iters = tf.constant(100)  # number of times to run batches
    with tf.GradientTape() as tape:
        tape.watch(win_reg)
        a1 = mdl.conv_layer(win_reg)
        a2 = mdl.flatten(a1)
        a3 = mdl.deep_1(a2)
        a4 = mdl.deep_2(a3)
        a5 = mdl.out(a4)

    # Calculate output-input gradient
    jacobian = tape.gradient(a5, win_reg)
    # Calculate less-approximated nlc numerator
    j1d = tf.reshape(jacobian, [jacobian.shape[0], jacobian.shape[1] * jacobian.shape[2]])
    # Calculate jacobian*Covx*jacobian^T
    j_covx_j = tf.stack([tf.linalg.trace(tf.matmul(v[None, :], tf.matmul(covx, v[:, None]))) for v in j1d])

    # Array building -- NLC numerator
    nlc_numerator_exact = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    nlc_numerator_approx = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for i in range(iters):
        # Randomly select input x
        inds_w_orep = random_choice(jacobian.shape[0] - tf.constant(train_frames), tf.multiply(btch, 2)) + tf.constant(
            train_frames)
        jc_jnorep = tf.gather(j_covx_j, inds_w_orep[:btch], axis=0)
        # Shove batches into array
        nlc_numerator_exact = nlc_numerator_exact.write(i, jc_jnorep)

        # Draw 'u' from n(0,1) distribution
        u = tf.Variable(tf.random.normal([btch]), dtype='float32')
        # Backpass 'u' through network with u^T[jacobian(x)]
        du_j = u[:, None, None] * tf.gather(jacobian, inds_w_orep[:btch], axis=0)
        # Draw randomly from input matrix, x'
        ran_xp = tf.gather(win_reg, inds_w_orep[btch:], axis=0)
        # the squared inner product of x' − x_bar and u^TJ(x)
        nlc_numerator_approx = nlc_numerator_approx.write(i,
                                                          tf.reshape(tf.reduce_sum(tf.multiply(du_j, ran_xp - avg_reg),
                                                                                   axis=(1, 2)) ** 2, btch))

    # One dimensionalize the numerators
    nlc_numerator_exact_1_d = tf.reshape(nlc_numerator_exact.stack(), (-1, 1))
    nlc_numerator_approx_1_d = tf.reshape(nlc_numerator_approx.stack(), (-1, 1))
    # THEE RESULT!!
    nlc = tf.reduce_mean(nlc_numerator_exact_1_d) / nlc_denominator
    nlc_approx = tf.reduce_mean(nlc_numerator_approx_1_d) / nlc_denominator

    return nlc[0], nlc_approx[0]


def random_choice(domain, sample, batch_shape=()) -> np.ndarray:
    """
  Generate random samples without replacement.
  This would be roughly equivalent to:
  numpy.random.choice(domain, sample, replace=False)
  but also supports generating batches of samples.
  """
    p = tf.random.uniform(batch_shape + (domain,), 0, 1)
    _, indices = tf.nn.top_k(p, sample)
    return indices
