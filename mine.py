"""
Module for easy running of MINE on user data
"""
import h5py
import numpy as np
from typing import List, Optional, Union
import utilities
import model
from taylorDecomp import all_decomposition_metrics, d2ca_dr2, data_mean_prediction
from perf_nlc_nonlin import calc_nlc
from dataclasses import dataclass


@dataclass
class MineData:
    """Class for the return values of MINE"""
    correlations_trained: np.ndarray
    correlations_test: np.ndarray
    taylor_scores: Optional[np.ndarray]
    taylor_true_change: Optional[np.ndarray]
    taylor_full_prediction: Optional[np.ndarray]
    taylor_by_predictor: Optional[np.ndarray]
    nl_probabilities: Optional[np.ndarray]
    mean_exp_scores: Optional[np.ndarray]
    jacobians: Optional[np.ndarray]

    def save_to_hdf5(self, file_object: Union[h5py.File, h5py.Group], overwrite=False) -> None:
        """
        Saves all contents to a hdf5 file or group object
        :param file_object: The file/group to save to
        :param overwrite: If true will overvwrite data in the file
        """
        utilities.create_overwrite(file_object, "correlations_trained", self.correlations_trained, overwrite)
        utilities.create_overwrite(file_object, "correlations_test", self.correlations_test, overwrite)
        if self.taylor_scores is not None:
            utilities.create_overwrite(file_object, "taylor_scores", self.taylor_scores, overwrite)
            utilities.create_overwrite(file_object, "taylor_true_change", self.taylor_true_change, overwrite)
            utilities.create_overwrite(file_object, "taylor_full_prediction", self.taylor_full_prediction, overwrite)
            utilities.create_overwrite(file_object, "taylor_by_predictor", self.taylor_by_predictor, overwrite)
        if self.nl_probabilities is not None:
            utilities.create_overwrite(file_object, "nl_probs", self.nl_probabilities, overwrite)
        if self.mean_exp_scores is not None:
            utilities.create_overwrite(file_object, "me_scores", self.mean_exp_scores, overwrite)
        if self.jacobians is not None:
            utilities.create_overwrite(file_object, "jacobians", self.jacobians, overwrite)


class Mine:
    """
    Class that collects intended model data and provides
    analysis function to be run on user-data
    """
    def __init__(self, train_fraction: float, model_history: int, corr_cut: float, compute_taylor: bool,
                 compute_complexity: bool, return_jacobians: bool, taylor_look_ahead: int, taylor_pred_every: int):
        """
        Create a new Mine object
        :param train_fraction: The fraction of frames to use for training 0 < train <= 1
        :param model_history: The number of frames to include in the model "history" (Note 1)
        :param corr_cut: Minimum correlation required on test data to compute other metrics
        :param compute_taylor: If true, perform taylor expansion and nonlinearity evaluation and return results
        :param compute_complexity: If true, compute mean-expansion scores and return results
        :param return_jacobians: If true, return the model jacobians at the data mean
        :param taylor_look_ahead: How many frames into the future to compute the taylor expansion (usually a few secs)
        :param taylor_pred_every: Every how many frames to compute the taylor expansion to save time
        """
        # Note 1: The ANN model purely looks into the past. However, the convolutional filters
        # can be centered arbitrarily by shifting predictor and response frames relative to
        # each other. See <processMusall.py> for an example
        if train_fraction <= 0 or train_fraction > 1:
            raise ValueError(f"train_fraction must be larger 0 and smaller or equal to 1 not {train_fraction}")
        self.train_fraction = train_fraction
        if model_history < 0:
            raise ValueError("model_history cant be < 0")
        self.model_history = model_history
        self.compute_taylor = compute_taylor
        self.compute_complexity = compute_complexity
        self.return_jacobians = return_jacobians
        self.n_epochs = 100  # sensible default
        self.taylor_look_ahead = taylor_look_ahead
        self.taylor_pred_every = taylor_pred_every
        self.corr_cut = corr_cut
        self.verbose = True

    def analyze_data(self, pred_data: List[np.ndarray], response_data: np.ndarray) -> MineData:
        """
        Process given data with MINE
        :param pred_data: Predictor data as a list of n_timepoints long vectors. Predictors are shared among all
            responses
        :param response_data: n_responses x n_timepoints matrix of responses
        :return:
            MineData object with the requested data
        """
        # check for matching sizes
        res_len = response_data.shape[1]
        for i, pd in enumerate(pred_data):
            if pd.size != res_len:
                raise ValueError(f"Predictor {i} has a different number of timesteps than the responses. {pd.size} vs. "
                                 f"{res_len}")
        # warn user if data is not standardized
        if not np.allclose(np.mean(response_data, 1), 0) or not np.allclose(np.std(response_data, 1), 1):
            print("WARNING: Response data does not appear standardized to 0 mean and standard deviation 1")
        for i, pd in enumerate(pred_data):
            if not np.isclose(np.mean(pd), 0, atol=1e-6) or not np.isclose(np.std(pd), 1, atol=1e-6):
                print(np.mean(pd))
                print(np.std(pd))
                print(f"WARNING: Predictor {i} does not appear standardized to 0 mean and standard deviation 1")
        train_frames = int(self.train_fraction * res_len)
        n_predictors = len(pred_data)

        correlations_trained = np.full(response_data.shape[0], np.nan)
        correlations_test = correlations_trained.copy()
        if self.compute_taylor:
            n_taylor = (n_predictors ** 2 - n_predictors) // 2 + n_predictors
            taylor_scores = np.full((response_data.shape[0], n_taylor, 2), np.nan)
            taylor_true_change = []
            taylor_full_prediction = []
            taylor_by_pred = []
            curvatures = correlations_test.copy()
            nlcs = correlations_test.copy()
        else:
            taylor_scores = None
            taylor_true_change = None
            taylor_full_prediction = None
            taylor_by_pred = None
            curvatures = None
            nlcs = None
        if self.compute_complexity:
            me_scores = correlations_test.copy()
        else:
            me_scores = None
        if self.return_jacobians:
            all_jacobians = np.full((response_data.shape[0], self.model_history * n_predictors), np.nan)
        else:
            all_jacobians = None

        data_obj = utilities.Data(self.model_history, pred_data, response_data, train_frames)
        # create model once
        m = model.get_standard_model(self.model_history)
        # the following is required to init variables at desired shape
        m(np.random.randn(1, self.model_history, len(data_obj.regressors)).astype(np.float32))
        # save untrained weights to reinitalize model without having to recreate the class which somehow leaks memory
        init_weights = m.get_weights()
        for cell_ix in range(data_obj.ca_responses.shape[0]):
            tset = data_obj.training_data(cell_ix, batch_size=256)
            regressors = data_obj.regressor_matrix(cell_ix)
            # reset weights to pre-trained state
            m.set_weights(init_weights)
            # the following appears to be required to re-init variables?
            m(np.random.randn(1, self.model_history, len(data_obj.regressors)).astype(np.float32))
            # train
            model.train_model(m, tset, self.n_epochs, data_obj.ca_responses.shape[1])
            # evaluate
            p, r = data_obj.predict_response(cell_ix, m)
            c_tr = np.corrcoef(p[:train_frames], r[:train_frames])[0, 1]
            correlations_trained[cell_ix] = c_tr
            c_ts = np.corrcoef(p[train_frames:], r[train_frames:])[0, 1]
            correlations_test[cell_ix] = c_ts
            # if the cell doesn't have a test correlation of at least corr_cut we skip the rest
            # NOTE: This means that some return values will only have one entry for each unit
            # that made the cut - the user will have to handle those cases
            if c_ts < self.corr_cut or not np.isfinite(c_ts):
                if self.verbose:
                    print(f"        Unit {cell_ix+1} out of {response_data.shape[0]} fit. "
                          f"Test corr={correlations_test[cell_ix]} which was below cut-off.")
                continue
            # compute taylor-expansion and nonlinearity evaluation if requested
            if self.compute_taylor:
                # compute taylor expansion
                true_change, pc, by_pred, avg_curve, allc = all_decomposition_metrics(m, regressors,
                                                                                      self.taylor_pred_every,
                                                                                      self.taylor_look_ahead)
                taylor_true_change.append(true_change)
                taylor_full_prediction.append(pc)
                taylor_by_pred.append(by_pred)
                # compute NLC
                nlc = calc_nlc(regressors, m)[0]
                curvatures[cell_ix] = avg_curve
                nlcs[cell_ix] = nlc
                # compute our by-predictor taylor importance as the fractional loss of r2 when excluding the component
                off_diag_index = 0
                for row in range(n_predictors):
                    for column in range(n_predictors):
                        if row == column:
                            remainder = pc - by_pred[:, row, column]
                            # Store in the first n_diag indices of taylor_by_pred (i.e. simply at row as indexer)
                            bsample = utilities.bootstrap_fractional_r2loss(true_change, pc, remainder, 1000)
                            taylor_scores[cell_ix, row, 0] = np.mean(bsample)
                            taylor_scores[cell_ix, row, 1] = np.std(bsample)
                        elif row < column:
                            remainder = pc - by_pred[:, row, column] - by_pred[:, column, row]
                            # Store in row-major order in taylor_by_pred after the first n_diag indices
                            bsample = utilities.bootstrap_fractional_r2loss(true_change, pc, remainder, 1000)
                            taylor_scores[cell_ix, n_predictors + off_diag_index, 0] = np.mean(bsample)
                            taylor_scores[cell_ix, n_predictors + off_diag_index, 1] = np.std(bsample)
                            off_diag_index += 1
            if self.compute_complexity or self.return_jacobians:
                # compute average predictor
                tset = data_obj.training_data(cell_ix, 256)
                all_inputs = []
                for inp, outp in tset:
                    all_inputs.append(inp.numpy())
                x_bar = np.mean(np.vstack(all_inputs), 0, keepdims=True)
                jacobian, hessian = d2ca_dr2(m, x_bar)
                if self.compute_complexity:
                    model_prediction, d_mean_prediction = data_mean_prediction(m, x_bar, jacobian, hessian, regressors,
                                                                               self.taylor_pred_every)
                    mean_pred_corr = np.corrcoef(model_prediction, d_mean_prediction)[0, 1]
                    me_scores[cell_ix] = mean_pred_corr**2
                if self.return_jacobians:
                    jacobian = jacobian.numpy().ravel()
                    # reorder jacobian by n_predictor long chunks of hist_steps timeslices
                    jacobian = np.reshape(jacobian, (self.model_history, n_predictors)).T.ravel()
                    all_jacobians[cell_ix, :] = jacobian
            if self.verbose:
                print(f"        Unit {cell_ix+1} out of {response_data.shape[0]} completed. "
                      f"Test corr={correlations_test[cell_ix]}")
        # convert nonlinearity metrics into probability if requested
        if self.compute_taylor:
            nonlin_lrm = utilities.NonlinClassifier.get_standard_model()
            nl_probs = nonlin_lrm.nonlin_probability(curvatures, nlcs)
            # turn the taylor predictions into ndarrays unless no unit passed threshold
            if len(taylor_true_change) > 0:
                if data_obj.ca_responses.shape[0] > 1:
                    taylor_true_change = np.vstack(taylor_true_change)
                    taylor_full_prediction = np.vstack(taylor_full_prediction)
                    taylor_by_pred = np.vstack([pbp[None, :] for pbp in taylor_by_pred])
                else:
                    # only one fit object, just expand dimension to keep things consistent
                    taylor_true_change = taylor_true_change[0][None, :]
                    taylor_full_prediction = taylor_full_prediction[0][None, :]
                    taylor_by_pred = taylor_by_pred[0][None, :]
            else:
                taylor_true_change = np.nan
                taylor_full_prediction = np.nan
                taylor_by_pred = np.nan
        else:
            nl_probs = None
        return_data = MineData(correlations_trained, correlations_test, taylor_scores, taylor_true_change,
                               taylor_full_prediction, taylor_by_pred, nl_probs, me_scores, all_jacobians)
        return return_data
