"""
Module for decomposing the prediction ANN into piecewise linear functions
via Taylor Series decomposition
"""

import model
import numpy as np
import tensorflow as tf
from numba import njit
from typing import List, Tuple


# def _dCa_dR_relu(mdl: model.ActivityPredictor, reg_input: np.ndarray) -> np.ndarray:
#     """
#     For the output unit of the ANN with relu activations computes the vector of first derivatives
#     with respect to each regressor input
#     :param mdl: The model for which to compute the derivative
#     :param reg_input: The input at which to compute the derivative
#     :returns: (input_length x nRegressors) long vector of partial first derivatives
#     """
#     # For notes see tanh implementation. The only difference here is the calculation of the outer derivatives
#     # in the chain rule: This derivative for ReLu is 0 whenever the activation is < 0, 1 whenever the activation
#     # is > 0 and undefined at 0 - to avoid NaN's we do what everyone incorrectly does and set it to 0 at 0
#
#     # CONVOLUTIONAL LAYER #
#     c = mdl.conv_layer(reg_input)  # (1 x 1 x N_c)
#     c = mdl.flatten(c).numpy()  # (1 x N_c)
#     weights_r_c = mdl.conv_layer.weights[0].numpy()  # (input_length x n_channels x N_c)
#     # linearize convolutional weights with respect to channel dimension so we later don't have
#     # to deal with matrices of partial derivatives but rather a big vector where groups of input_length
#     # elements belong to one given regressor
#     weights_r_c = weights_r_c.reshape((weights_r_c.shape[0]*weights_r_c.shape[1], weights_r_c.shape[2]))
#     dc_dr = weights_r_c  # (N_r x N_c)
#
#     # FIRST DEEP LAYER #
#     x = mdl.deep_1(c).numpy()
#     weights_c_x = mdl.deep_1.weights[0].numpy()  # (N_c x N_x)
#     d_relu_x = np.diag((x.ravel() > 0).astype(np.float32))
#     dx_dr = np.matmul(dc_dr, np.matmul(d_relu_x, weights_c_x.T).T)  # (N_r x N_x)
#
#     # SECOND DEEP LAYER #
#     y = mdl.deep_2(x).numpy()
#     weights_x_y = mdl.deep_2.weights[0].numpy()  # (N_x x N_y)
#     d_relu_y = np.diag((y.ravel() > 0).astype(np.float32))
#     dy_dr = np.matmul(dx_dr, np.matmul(d_relu_y, weights_x_y.T).T)  # (N_r x N_y)
#
#     # OUTPUT NODE #
#     weights_y_z = mdl.out.weights[0].numpy()  # (N_y x 1)
#     dz_dr = np.matmul(dy_dr, weights_y_z)  # (N_r x 1)
#     return dz_dr.ravel()
#
#
# def _ddCa_dRdR_relu(mdl: model.ActivityPredictor, reg_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     For the output unit of the ANN with relu activations computes the matrix of second derivatives
#     with respect to each regressor input
#     :param mdl: The model for which to compute the derivative
#     :param reg_input: The input at which to compute the derivative
#     :returns:
#         [0] (input_length x nRegressors) long vector of partial first derivatives
#         [1] (input_length x nRegressors) x (input_length x nRegressors) matrix of partial second derivatives
#     """
#     # With relu activation all second partial derivatives are 0 (or technically undefined for
#     # inputs into a node = 0)
#     return _dCa_dR_relu(mdl, reg_input), np.zeros((reg_input.size, reg_input.size), np.float32)
#
#
# def _dCa_dR_tanh(mdl: model.ActivityPredictor, reg_input: np.ndarray) -> np.ndarray:
#     """
#     For the output unit of the ANN with tanh activations computes the vector of first derivatives
#     with respect to each regressor input
#     :param mdl: The model for which to compute the derivative
#     :param reg_input: The input at which to compute the derivative
#     :returns: (input_length x nRegressors) long vector of partial first derivatives
#     """
#     # r: Regressor inputs
#     # c: Convolutional layer output (Note: This layer is linear)
#     # x: First deep layer output
#     # y: Second deep layer output
#     # z: Output (one node, linear)
#     # N_r: Total length of regressor input = =n_regressors x input_length
#     # N_c: Number of convolutional nodes
#     # N_x: Number of deep layer 1 nodes
#     # N_y: Number of deep layer 2 nodes
#     # weigths_r_c: weights from layer R to layer C, etc.
#     # dc_dr: derivative of convolutional layer units with respect to regressor inputs, etc.
#     # shape notes the number of rows followed by number of columns in brackets
#
#     # CONVOLUTIONAL LAYER #
#     c = mdl.conv_layer(reg_input)  # (1 x 1 x N_c)
#     c = mdl.flatten(c).numpy()  # (1 x N_c)
#     weights_r_c = mdl.conv_layer.weights[0].numpy()  # (input_length x n_channels x N_c)
#     # linearize convolutional weights with respect to channel dimension so we later don't have
#     # to deal with matrices of partial derivatives but rather a big vector where groups of n_channels
#     # elements belong to one given timepoint
#     weights_r_c = weights_r_c.reshape((weights_r_c.shape[0]*weights_r_c.shape[1], weights_r_c.shape[2]))
#     dc_dr = weights_r_c  # (N_r x N_c)
#
#     # FIRST DEEP LAYER #
#     x = mdl.deep_1(c).numpy()  # already has nonlinearity applied!!!
#     # this is the activation before the nonlinearity
#     xa = tf.add(tf.matmul(c, mdl.deep_1.weights[0]), mdl.deep_1.weights[1]).numpy()
#     weights_c_x = mdl.deep_1.weights[0].numpy()  # (N_c x N_x)
#     d_tanh_x = np.diag(((1 / np.cosh(xa)) ** 2).ravel()) since dtanh/dx = sech2(x) and sech(x) = 1/cosh(x) (N_x x N_x)
#     dx_dr = np.matmul(dc_dr, np.matmul(d_tanh_x, weights_c_x.T).T)  # (N_r x N_x)
#
#     # SECOND DEEP LAYER #
#     ya = tf.add(tf.matmul(x, mdl.deep_2.weights[0]), mdl.deep_2.weights[1]).numpy()
#     weights_x_y = mdl.deep_2.weights[0].numpy()  # (N_x x N_y)
#     d_tanh_y = np.diag(((1 / np.cosh(ya)) ** 2).ravel())  # (N_y x N_y)
#     dy_dr = np.matmul(dx_dr, np.matmul(d_tanh_y, weights_x_y.T).T)  # (N_r x N_y)
#
#     # OUTPUT NODE #
#     weights_y_z = mdl.out.weights[0].numpy()  # (N_y x 1)
#     dz_dr = np.matmul(dy_dr, weights_y_z)  # (N_r x 1)
#     return dz_dr.ravel()
#
#
# @njit
# def _compute_part_tensor(projection: np.ndarray, n: int):
#     """
#     Computes intermediate tensor for second derivative computations
#     :param projection: Projection of previous layer derivative by current layer weights
#     :param n: Total number of input elements (n_timepoints x n_regressors)
#     :return: The intermediate tensor for computation
#     """
#     part_tensor = np.empty((n, projection.shape[1], n), dtype=np.float32)
#     for r1 in range(n):
#         for r2 in range(r1, n):
#             val = projection[r1, :] * projection[r2, :]
#             part_tensor[r1, :, r2] = val
#             part_tensor[r2, :, r1] = val
#     return part_tensor
#
#
# def _ddCa_dRdR_tanh(mdl: model.ActivityPredictor, reg_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     For the output unit of the ANN with tanh activations computes the matrix of second and vector of
#     first derivatives with respect to each regressor input
#     :param mdl: The model for which to compute the derivative
#     :param reg_input: The input at which to compute the derivative
#     :returns:
#         [0] (input_length x nRegressors) long vector of partial first derivatives
#         [1] (input_length x nRegressors) x (input_length x nRegressors) matrix of partial second derivatives
#     """
#     # CONVOLUTIONAL LAYER #
#     c = mdl.conv_layer(reg_input)  # (1 x 1 x N_c)
#     c = mdl.flatten(c).numpy()  # (1 x N_c)
#     weights_r_c = mdl.conv_layer.weights[0].numpy()  # (input_length x n_channels x N_c)
#     # linearize convolutional weights with respect to channel dimension so we later don't have
#     # to deal with matrices of partial derivatives but rather a big vector where groups of N-regressor
#     # elements belong to one given timepoint
#     weights_r_c = weights_r_c.reshape((weights_r_c.shape[0]*weights_r_c.shape[1], weights_r_c.shape[2]))
#     dc_dr = weights_r_c  # (N_r x N_c)
#     # second derivative is 0 in a linear layer
#     # ddc_drr = np.zeros((weights_r_c.shape[0], c.size, weights_r_c.shape[0]), dtype=np.float32)  # (N_r x N_c x N_r)
#
#     # FIRST DEEP LAYER #
#     # the following is the activation before the nonlinearity
#     weights_c_x = mdl.deep_1.weights[0].numpy()  # (N_c x N_x)
#     bias_x = mdl.deep_1.weights[1].numpy()
#     xa = c@weights_c_x + bias_x  # activation before the nonlinearity
#     x = np.tanh(xa)  # output of the neurons
#     d_tanh_x = np.diag((1 - x**2).ravel())  # since dtanh/dx = sech2(x) & sech2(x) = 1-tanh2(x) (N_x x N_x)
#     dd_tanh_x = -2 * d_tanh_x * np.diag(x.ravel())  # since ddtanh/dx = -2 * dtanh/dx * tanh (N_x x N_x)
#     dx_dr = dc_dr@(d_tanh_x@weights_c_x.T).T  # (N_r x N_x)
#     dc_dr_projection = dc_dr@weights_c_x  # (N_r x N_x)
#     # NOTE: The following part_tensor is independent of the input, in other words it can be calculated once per model
#     if mdl.part_tensor_1 is not None:
#         part_tensor_1 = mdl.part_tensor_1
#     else:
#         part_tensor_1 = _compute_part_tensor(dc_dr_projection, weights_r_c.shape[0])
#         mdl.part_tensor_1 = part_tensor_1
#     # Note: for symmetry we could compute the first term here, but it is by definition 0
#     # ddx_drr = np.matmul(d_tanh_x, np.matmul(weights_c_x.T, ddc_drr)) + np.matmul(dd_tanh_x, part_tensor)
#     ddx_drr = dd_tanh_x@part_tensor_1
#
#     # SECOND DEEP LAYER #
#     weights_x_y = mdl.deep_2.weights[0].numpy()  # (N_x x N_y)
#     bias_y = mdl.deep_2.weights[1].numpy()
#     ya = x@weights_x_y + bias_y
#     y = np.tanh(ya)
#     d_tanh_y = np.diag((1 - y**2).ravel())  # (N_y x N_y)
#     dd_tanh_y = -2 * d_tanh_y * np.diag(y.ravel())  # (N_x x N_x)
#     dy_dr = dx_dr@(d_tanh_y@weights_x_y.T).T  # (N_r x N_y)
#     dx_dr_projection = dx_dr@weights_x_y  # (N_r x N_y)
#     part_tensor_2 = _compute_part_tensor(dx_dr_projection, weights_r_c.shape[0])
#     ddy_drr = d_tanh_y@(weights_x_y.T@ddx_drr) + dd_tanh_y@part_tensor_2
#
#     # OUTPUT NODE #
#     weights_y_z = mdl.out.weights[0].numpy()  # (N_y x 1)
#     dz_dr = dy_dr@weights_y_z  # (N_r x 1)
#     ddz_drr = weights_y_z.T@ddy_drr  # (N_r x 1 x N_r)
#
#     return dz_dr.ravel(), ddz_drr[:, 0, :]


@tf.function
def d2ca_dr2(mdl: model.ActivityPredictor, reg_input: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
    x = tf.convert_to_tensor(reg_input)
    with tf.GradientTape() as t2:
        t2.watch(x)
        with tf.GradientTape() as t1:
            t1.watch(x)
            # NOTE: The following is slightly faster than ca = mdl(x) presumably due to skipping of dropout layers
            c = mdl.conv_layer(x)
            d1 = mdl.deep_1(mdl.flatten(c))
            d2 = mdl.deep_2(d1)
            ca = mdl.out(d2)
        jacobian = t1.gradient(ca, x)
    hessian = t2.jacobian(jacobian, x)
    return jacobian, hessian


@tf.function
def dca_dr(mdl: model.ActivityPredictor, reg_input: np.ndarray) -> tf.Tensor:
    x = tf.convert_to_tensor(reg_input)
    with tf.GradientTape() as t1:
        t1.watch(x)
        # NOTE: The following is slightly faster than ca = mdl(x) presumably due to skipping of dropout layers
        c = mdl.conv_layer(x)
        d1 = mdl.deep_1(mdl.flatten(c))
        d2 = mdl.deep_2(d1)
        ca = mdl.out(d2)
    jacobian = t1.gradient(ca, x)
    return jacobian


def taylor_predict(mdl: model.ActivityPredictor, regressors: np.ndarray, use_d2: bool, take_every: int,
                   predict_ahead=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each time t in regressors, evaluates the model, computes the selected derivatives and
    then attempts to predict the model response at time t+1
    :param mdl: The model to use for predictions
    :param regressors: The 2D regressor matrix, n_timesteps x m_regressors
    :param use_d2: If set to False only the first derivative will be used for the prediction
    :param take_every: Only form predictions every n frames to save time
    :param predict_ahead: The number of frames to predict ahead with the taylor expansion
    :returns:
        [0]: (n_timesteps-input_length-predict_ahead)/n long timeseries of taylor predictions
        [1]: (n_timesteps-input_length-predict_ahead)/n long timeseries of actual network outputs
    """
    if predict_ahead < 1:
        raise ValueError("predict_ahead has to be integer >= 1")
    inp_length = mdl.input_length
    mdl_output = []
    taylor_prediction = []
    t = inp_length - 1
    while t < regressors.shape[0]-predict_ahead:
        cur_regs = regressors[None, t - inp_length + 1: t + 1, :]
        next_regs = regressors[None, t - inp_length + 1 + predict_ahead: t + 1 + predict_ahead, :]
        cur_mod_out = mdl.get_output(cur_regs)
        next_mod_out = mdl.get_output(next_regs)
        mdl_output.append(next_mod_out)
        if use_d2:
            d1, d2 = d2ca_dr2(mdl, cur_regs)
            d1 = d1.numpy().ravel()
            d2 = np.reshape(d2.numpy(), (regressors.shape[1] * mdl.input_length,
                                         regressors.shape[1] * mdl.input_length))
        else:
            d1 = dca_dr(mdl, cur_regs)
            d1 = d1.numpy().ravel()
            d2 = None
        tay_pred = _taylor_predict(cur_regs, next_regs, cur_mod_out, d1, d2)
        taylor_prediction.append(tay_pred)
        t += take_every
    return np.hstack(taylor_prediction), np.hstack(mdl_output)


def _taylor_predict(reg_fix_point: np.ndarray, reg_test: np.ndarray, ann_fix: float, d1: np.ndarray,
                    d2: np.ndarray) -> float:
    """
    Computes the taylor prediction about a point for another test point nearby
    :param reg_fix_point: The regressor input at the fix point where derivatives have been calculated
    :param reg_test: The regressor input for which to predict the ann response
    :param ann_fix: The output of the ann at reg_fix_point
    :param d1: The set of first order partial derivatives at reg_fix_point
    :param d2: The matrix of second order partial derivatives at reg_fix_point
    """
    diff = (reg_test - reg_fix_point).ravel()
    if d2 is None:
        return ann_fix + np.dot(diff, d1)
    return ann_fix + np.dot(diff, d1) + 0.5 * np.sum(np.dot(diff[:, None], diff[None, :]) * d2)


@njit
def _compute_taylor_d2(reg_diff: np.ndarray, d2: np.ndarray, nregs: int, inp_length: int) -> np.ndarray:
    """
    Computes responses belonging to the second derivative, rearranging by regressor
    instead of by time
    :param reg_diff: The difference in regressors as 2D (1 x (n_regs*n_timepoints)) vector
    :param d2: The hessian
    :param nregs: The number of regressors
    :param inp_length: The timelength of each regressor input
    :return: The second derivative contribution ((n_regs*n_timepoints) x (n_regs*n_timepoints))
    """
    taylor_d2_temp = 0.5 * np.dot(reg_diff, reg_diff.T) * d2  # this matrix is organized by time not by regressor
    taylor_d2 = np.empty_like(taylor_d2_temp, dtype=np.float32)
    for row in range(taylor_d2_temp.shape[0]):
        regnum = row % nregs
        time = row // nregs
        row_ix = regnum * inp_length + time
        for col in range(taylor_d2_temp.shape[1]):
            regnum = col % nregs
            time = col // nregs
            col_ix = regnum * inp_length + time
            taylor_d2[row_ix, col_ix] = taylor_d2_temp[row, col]
    return taylor_d2


@njit
def _compute_by_reg(taylor_d1: np.ndarray, taylor_d2: np.ndarray, nregs: int, inp_length: int) -> np.ndarray:
    """
    Aggregates derivative contributions by regressor
    :param taylor_d1: The first partial derivative contributions (by regressor and time vector)
    :param taylor_d2: The second partial derivative contributions (by regressor and time square matrix)
    :param nregs: The number of regressors
    :param inp_length: The number of timepoints
    :return: Contribution aggregated by regressor as array (1 xnregs x nregs) to account for possible interactions
    """
    by_reg = np.full((1, nregs, nregs), 0.0, dtype=np.float32)
    for r1 in range(nregs):
        for r2 in range(nregs):
            if r1 == r2:
                # these are the non-interacting parts which need to take d1 into account
                by_reg[0, r1, r2] += np.sum(taylor_d1[r1 * inp_length:(r1 + 1) * inp_length])
            by_reg[0, r1, r2] += np.sum(
                taylor_d2[r1 * inp_length:(r1 + 1) * inp_length, r2 * inp_length:(r2 + 1) * inp_length])
    return by_reg


def taylor_decompose(mdl: model.ActivityPredictor, regressors: np.ndarray, take_every: int, predict_ahead: int,
                     use_d2=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Uses taylor decomposition to predict changes in network output around chosen point using the
    all information as well as only the information corresponding to each regressor and their
    interactions terms
    :param mdl: The model to use for predictions
    :param regressors: The 2D regressor matrix, n_timesteps x m_regressors
    :param take_every: Only form predictions every n frames to save time
    :param predict_ahead: The number of frames to predict ahead with the taylor expansion
    :param use_d2: If set to false only the first derivative will be used in the taylor expansion
    :returns:
        [0]: The true change for each timepoint by going predict_ahead frames into the future
                (n_timesteps-input_length-predict_ahead)/n long vector
        [1]: The predicted change for the whole taylor series
                (n_timesteps-input_length-predict_ahead)/n long vector
        [2]: Array of predicted changes by regressors and their interactions
                (n_timesteps-input_length-predict_ahead)/n x n_regressors x n_regressors
    """
    if predict_ahead < 1:
        raise ValueError("predict_ahead has to be integer >= 1")
    inp_length = mdl.input_length
    mdl_out_change = []
    full_tp_change = []
    by_reg_tp_change = []
    t = inp_length - 1
    nregs = regressors.shape[1]
    while t < regressors.shape[0]-predict_ahead:
        cur_regs = regressors[None, t - inp_length + 1: t + 1, :]
        next_regs = regressors[None, t - inp_length + 1 + predict_ahead: t + 1 + predict_ahead, :]
        reg_diff = (next_regs - cur_regs).ravel().astype(np.float64)
        cur_mod_out = mdl.get_output(cur_regs)
        next_mod_out = mdl.get_output(next_regs)
        mdl_out_change.append(next_mod_out - cur_mod_out)
        # unfortunately, none of the following are contiguous by regressor but rather by timepoint
        # we therefore need to reshape taylor_d1 and taylor_d2 to allow for easy indexing below
        if use_d2:
            d1, d2 = d2ca_dr2(mdl, cur_regs)
            d1 = d1.numpy().ravel()
            d2 = np.reshape(d2.numpy(), (regressors.shape[1] * mdl.input_length,
                                         regressors.shape[1] * mdl.input_length))
            taylor_d1 = reg_diff * d1.astype(np.float64)
            taylor_d1 = np.reshape(taylor_d1, (inp_length, nregs)).T.ravel()
            taylor_d2 = _compute_taylor_d2(reg_diff[:, None], d2.astype(np.float64), nregs, inp_length)
        else:
            d1 = dca_dr(mdl, cur_regs)
            d1 = d1.numpy().ravel()
            taylor_d1 = reg_diff * d1.astype(np.float64)
            taylor_d1 = np.reshape(taylor_d1, (inp_length, nregs)).T.ravel()
            taylor_d2 = np.zeros((taylor_d1.size, taylor_d1.size))
        full_tp_change.append(np.sum(taylor_d1) + np.sum(taylor_d2))
        by_reg = _compute_by_reg(taylor_d1, taylor_d2, nregs, inp_length)
        by_reg_tp_change.append(by_reg)
        t += take_every
    return np.hstack(mdl_out_change), np.hstack(full_tp_change), np.vstack(by_reg_tp_change)


# TODO: We should really harmonize the argument order between taylor_decompose and avg_directional_curvature!!
def avg_directional_curvature(mdl: model.ActivityPredictor, regressors: np.ndarray, delta_distance: int,
                              take_every: int) -> Tuple[float, List[float]]:
    """
    Along a trajectory through input space defined by regressors computes the average
    curvature of the function approximated by the ANN
    :param mdl: The model to use for predictions
    :param regressors: The 2D regressor matrix, n_timesteps x m_regressors
    :param delta_distance: How many timesteps to look ahead to compute direction in input space
    :param take_every: Only curvature every n frames to save time
    :return:
        [0]: Scalar, the average curvature along the path
        [1]: List of all point-wise curvatures
    """
    if delta_distance < 1:
        raise ValueError("delta_distance has to be integer >= 1")
    inp_length = mdl.input_length
    t = inp_length - 1
    all_curvatures = []
    while t < regressors.shape[0]-delta_distance:
        cur_regs = regressors[None, t - inp_length + 1: t + 1, :]
        next_regs = regressors[None, t - inp_length + 1 + delta_distance: t + 1 + delta_distance, :]
        d2 = d2ca_dr2(mdl, cur_regs)[1]
        if d2 is None:
            d2 = np.zeros((regressors.shape[1] * mdl.input_length,
                           regressors.shape[1] * mdl.input_length))
        else:
            d2 = np.reshape(d2.numpy(), (regressors.shape[1] * mdl.input_length,
                                         regressors.shape[1] * mdl.input_length))
        dir_curv = _directional_curvature(cur_regs, next_regs, d2)
        all_curvatures.append(dir_curv)
        t += take_every
    avg_curve = float(np.mean(all_curvatures))
    return avg_curve, all_curvatures


def _directional_curvature(reg_fix_point: np.ndarray, reg_ahead: np.ndarray, d2: np.ndarray) -> float:
    """
    Computes the local curvature in direction of movement in the input space
    from reg_fix_point to reg_ahead
    :param reg_fix_point: The regressor input at the current timepoint
    :param reg_ahead: The regressor input at the next point on the path through input space
    :param d2: The hessian at the current timepoint
    :return: The absolute curvature at fix-point in path direction = ||(reg_ahead-reg_fix_point)d2||2
    """
    diff = (reg_ahead - reg_fix_point).ravel()[None, :]
    nm = np.linalg.norm(diff)
    if nm == 0:
        nm = 10
    return np.linalg.norm(np.dot(diff, d2)) / nm


def all_decomposition_metrics(mdl: model.ActivityPredictor, regressors: np.ndarray, take_every: int,
                              look_ahead: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Computes both the taylor decomposition and non-linearity metric to save time by calculating the needed
    derivatives only once
    :param mdl: The model to use for predictions
    :param regressors: The 2D regressor matrix, n_timesteps x m_regressors
    :param take_every: Only compute metrics every n frames to save time
    :param look_ahead: How many timesteps to look ahead to compute direction in input space
    :return:
        [0]: The true change for each timepoint by going predict_ahead frames into the future
                (n_timesteps-input_length-predict_ahead)/n long vector
        [1]: The predicted change for the whole taylor series
                (n_timesteps-input_length-predict_ahead)/n long vector
        [2]: Array of predicted changes by regressors and their interactions
                (n_timesteps-input_length-predict_ahead)/n x n_regressors x n_regressors
        [3]: Scalar, the average non-linearity metric along the path
        [4]: Array of point-wise non-linearity metrics
    """
    if look_ahead < 1:
        raise ValueError("look_ahead has to be integer >= 1")
    inp_length = mdl.input_length
    mdl_out_change: List[float] = []
    full_tp_change: List[float] = []
    by_reg_tp_change: List[np.ndarray] = []
    all_curvatures: List[float] = []
    t: int = inp_length - 1
    nregs: int = regressors.shape[1]
    while t < regressors.shape[0]-look_ahead:
        cur_regs = regressors[None, t - inp_length + 1: t + 1, :]
        next_regs = regressors[None, t - inp_length + 1 + look_ahead: t + 1 + look_ahead, :]
        reg_diff = (next_regs - cur_regs).ravel().astype(np.float64)
        cur_mod_out = mdl.get_output(cur_regs)
        next_mod_out = mdl.get_output(next_regs)
        mdl_out_change.append(next_mod_out - cur_mod_out)
        d1, d2 = d2ca_dr2(mdl, cur_regs)
        d1 = d1.numpy().ravel()
        d2 = np.reshape(d2.numpy(), (regressors.shape[1] * mdl.input_length,
                                     regressors.shape[1] * mdl.input_length))
        # compute non-linearity metric
        dir_curv = _directional_curvature(cur_regs, next_regs, d2)
        all_curvatures.append(dir_curv)
        # compute taylor decomposition - note, since the derivatives are not orderd by regressor but by time
        # they need to be re-arranged
        taylor_d1 = reg_diff * d1.astype(np.float64)
        taylor_d1 = np.reshape(taylor_d1, (inp_length, nregs)).T.ravel()
        taylor_d2 = _compute_taylor_d2(reg_diff[:, None], d2.astype(np.float64), nregs, inp_length)
        full_tp_change.append(np.sum(taylor_d1) + np.sum(taylor_d2))
        by_reg = _compute_by_reg(taylor_d1, taylor_d2, nregs, inp_length)
        by_reg_tp_change.append(by_reg)
        t += take_every
    avg_curve = float(np.mean(all_curvatures))
    return np.hstack(mdl_out_change), np.hstack(full_tp_change), np.vstack(by_reg_tp_change), avg_curve,\
        np.hstack(all_curvatures)


def data_mean_prediction(mdl: model.ActivityPredictor, x_bar, j_x_bar, h_x_bar, regressors: np.ndarray,
                         take_every: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the prediction of responses based on a fixed Taylor expansion of the network around a specific point
    in our case taken to be the data mean
    :param mdl: The CNN model
    :param x_bar: The data mean (or any arbitrary fix point)
    :param j_x_bar: The jacobian of the model at x_bar
    :param h_x_bar: The hession of the model at x_bar
    :param regressors: The 2D regressor matrix, n_timesteps x m_regressors
    :param take_every: Only compute metrics every n frames to save time
    :return:
        [0]: The prediction of the CNN model
        [1]: The prediction of the fixed-point expansion
    """
    f_x_bar = mdl(x_bar)
    mean_prediction: List[float] = []
    mdl_out_change: List[float] = []
    inp_length = mdl.input_length
    t: int = inp_length - 1
    # prepare our first and second derivatives at the data mean
    d1 = j_x_bar.numpy().ravel()
    d2 = np.reshape(h_x_bar.numpy(), (regressors.shape[1] * mdl.input_length, regressors.shape[1] * mdl.input_length))
    while t < regressors.shape[0]:
        cur_regs = regressors[None, t - inp_length + 1: t + 1, :]
        # compute our difference to the data mean
        reg_diff = (cur_regs - x_bar).ravel().astype(np.float64)
        # get the actual model prediction at the current point and add to our return
        cur_mod_out = mdl.get_output(cur_regs)
        mdl_out_change.append(cur_mod_out)
        # compute taylor decomposition around data mean
        mp = f_x_bar + np.dot(reg_diff, d1) + 0.5 * np.sum(np.dot(reg_diff[:, None], reg_diff[None, :]) * d2)
        mean_prediction.append(mp)
        t += take_every
    return np.hstack(mdl_out_change), np.hstack(mean_prediction)


if __name__ == "__main__":
    print("Module for ANN Taylor decomposition")
