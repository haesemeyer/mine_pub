
"""
Module with data preparation classes and functions
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
import model
import h5py
from typing import Union, List, Any, Optional
from scipy import signal as sig
from scipy.stats import entropy
from sklearn.neighbors import KDTree
from sklearn.cluster import SpectralClustering
from numba import njit


def spectral_cluster(in_data: np.ndarray, n_clusters: int, min_clust_cosim: float, cosim_cut=0.2) -> np.ndarray:
    """
    Performs spectral clustering with subsequent correlation based tightening on the data
    :param in_data: n_samples x n_features matrix of input data
    :param n_clusters: The number of clusters to form
    :param min_clust_cosim: The cosine similarity each trace has to have to the cluster centroid to be included
    :param cosim_cut: The cut of the cosine similarity matrix fed into the spectral clustering (cosim<cut -> 0) >=0
    :return: n_samples long vector of cluster memberships (-1 for not clustered)
    """
    if cosim_cut < 0:
        raise ValueError("Cosine similarity cut has to be >= 0")
    traces = in_data / np.linalg.norm(in_data, axis=1, keepdims=True)
    membership = np.full(traces.shape[0], -1, dtype=int)
    # create pairwise cosine similarity matrix
    cosim_mat = traces @ traces.T
    cosim_mat[cosim_mat < cosim_cut] = 0
    # cluster based on cosine similarity matrix
    spc = SpectralClustering(n_clusters, affinity="precomputed")
    spc.fit(cosim_mat)
    initial_labels = spc.labels_
    # extract cluster averages for angle-based refinement
    cluster_avgs = np.zeros((n_clusters, traces.shape[1]))
    for i in range(n_clusters):
        if np.sum(initial_labels == i) > 1:
            cluster_avgs[i, :] = np.mean(traces[initial_labels == i, :], 0)
        else:
            cluster_avgs[i, :] = traces[initial_labels == i, :]
    cluster_avgs /= np.linalg.norm(cluster_avgs, axis=1, keepdims=True)
    # calculate cosine similarity of each trace to each cluster centroid and assign to either max correlation or -1
    for i, tr in enumerate(traces):
        cl_num = -1
        c_max = -1
        for j in range(n_clusters):
            c = np.dot(tr, cluster_avgs[j])
            if c > min_clust_cosim and c > c_max:
                cl_num = j
                c_max = c
        membership[i] = cl_num
    return membership


def greedy_cosine_cluster(traces: np.ndarray, c_size_cut: int, min_sim: float, min_auto_merge=0.98) -> np.ndarray:
    """
    Performs clustering similar to Bianco 2015. Clusters are formed by pairwise correlations adding highest correlated
    pairs into clusters.
    :param traces: n_samples x n_features matrix of activity data
    :param c_size_cut: Clusters with a size below this size will be dropped
    :param min_sim: The minimal cosine similarity for merging units to a cluster / other units
    :param min_auto_merge: The minimum correlation for merging without recalculating cluster averages (default ~10 deg)
    :return: n_samples long vector of cluster assignments
    """
    def make_cos_mat(t: np.ndarray):
        """
        Creates a cosine similarity matrix that can be easily processed by our algorithm
        Assumes that all entries in t are unit length, n_samplesxn_features
        """
        cm = t @ t.T
        np.fill_diagonal(cm, -np.inf)
        cm[np.isnan(cm)] = -np.inf
        return cm

    traces = traces / np.linalg.norm(traces, axis=1, keepdims=True)  # turn into unit-vectors
    clust_assign = np.arange(traces.shape[0])  # each unit is it's own valid (!= -1) cluster
    full_index = clust_assign.copy()  # to translate indices
    x_working = traces.copy()  # in this matrix, individual traces will be replaced by cluster averages
    # the following tracks which samples should be used for forming the correlation matrix - as clusters are formed
    # only the cluster average would be used
    samples_used = np.full(traces.shape[0], True, dtype=bool)
    coss_mat = make_cos_mat(x_working[samples_used])
    n_calcs = 0
    while np.max(coss_mat) >= min_sim and np.sum(samples_used) > 1:
        print(
            f"Recalculated cosine matrix. {np.sum(samples_used) / samples_used.size * 100} % of samples not clustered")
        samples_used_this_round = samples_used.copy()
        while True:
            # find pair of samples or clusters to merge
            max_1, max_2 = np.unravel_index(np.argmax(coss_mat), coss_mat.shape)
            # get current cluster numbers belonging to these samples
            cn_max_1 = clust_assign[samples_used_this_round][max_1]
            cn_max_2 = clust_assign[samples_used_this_round][max_2]
            # assign smaller of the current cluster numbers to all involved units. Remove the other from samples used
            # and replace the value in x_working at the smaller cluster number with the new cluster average
            if cn_max_1 < cn_max_2:
                clust_assign[clust_assign == cn_max_2] = cn_max_1
                samples_used[full_index[samples_used_this_round][max_2]] = False  # is now clustered
                coss_mat[:, max_2] = -np.inf
                coss_mat[max_2, :] = -np.inf
                m = np.mean(traces[clust_assign == cn_max_1], 0)
                m /= np.linalg.norm(m)  # make sure that it is unit length upon insertion
                assert m.size == traces.shape[1]
                x_working[full_index[samples_used_this_round][max_1], :] = m
            elif cn_max_2 < cn_max_1:
                clust_assign[clust_assign == cn_max_1] = cn_max_2
                samples_used[full_index[samples_used_this_round][max_1]] = False
                coss_mat[:, max_1] = -np.inf
                coss_mat[max_1, :] = -np.inf
                m = np.mean(traces[clust_assign == cn_max_2], 0)
                m /= np.linalg.norm(m)  # make sure that it is unit length upon insertion
                assert m.size == traces.shape[1]
                x_working[full_index[samples_used_this_round][max_2], :] = m
            else:
                assert False
            if np.max(coss_mat) < min_auto_merge:
                # nothing matches our aut-merging threshold - therefore re-calcuate
                # correlation matrix with formed cluster means and then proceed
                break
        coss_mat = make_cos_mat(x_working[samples_used])
        n_calcs += 1
    print(f"Performed {n_calcs} calculations of the cosine similarity matrix")
    # clean up clusters
    clust_return = np.full(clust_assign.size, -1, dtype=int)
    all_cnums = np.unique(clust_assign)
    cn_to_assign = 0
    for cn in all_cnums:
        if np.sum(clust_assign == cn) >= c_size_cut:
            clust_return[clust_assign == cn] = cn_to_assign
            cn_to_assign += 1
    return clust_return


def create_overwrite(storage: Union[h5py.File, h5py.Group], name: str, data: Any, overwrite: bool,
                     compress=False) -> None:
    """
    Allows to create a new dataset in an hdf5 file an if desired overvwrite any old data
    :param storage: The hdf5 file or group used to store the information
    :param name: The name of the dataset
    :param data: The data
    :param overwrite: If true any old data with the same name will be deleted and subsequently replaced
    :param compress: If true, data will be stored compressed
    """
    if overwrite and name in storage:
        del storage[name]
    if compress:
        storage.create_dataset(name, data=data, compression="gzip", compression_opts=5)
    else:
        storage.create_dataset(name, data=data)


def create_random_wave_predictor(time_base: np.ndarray) -> np.ndarray:
    reg_sin = np.zeros_like(time_base)
    for i in range(5):
        freq = np.random.rand() / 40
        shift = (np.random.rand() - 0.5) * 5
        amp = (np.random.rand() * 0.25 + 0.5) / 5
        reg_sin += amp * np.sin(time_base * 2 * np.pi * freq + shift).astype(np.float32)
    freq = np.random.rand() / 40
    shift = (np.random.rand() - 0.5) * 5
    duty = np.random.rand() * 0.25 + 0.25
    amp = np.random.rand() * 0.25 + 0.5
    reg_square = amp * sig.square(time_base * 2 * np.pi * freq + shift, duty)
    freq = np.random.rand() / 40
    shift = (np.random.rand() - 0.5) * 5
    width = np.random.rand()
    amp = np.random.rand() * 0.25 + 0.5
    reg_triangle = amp * sig.sawtooth(time_base * 2 * np.pi * freq + shift, width)
    return reg_sin + reg_square + reg_triangle


def bootstrap_fractional_r2loss(real: np.ndarray, predicted: np.ndarray, remainder: np.ndarray,
                                n_boot: int) -> np.ndarray:
    """
    Returns bootstrap samples for the loss in r^2 after components are removed from a prediction
    :param real: The real timeseries data
    :param predicted: The full prediction for the timeseries
    :param remainder: The prediction for the timeseries after components have been excluded
    :param n_boot:
    :return: n_boot long vector of fractional loss scores (fraction of r2 that is lost going from full to remainder)
    """
    if real.size != predicted.size or predicted.size != remainder.size:
        raise ValueError("All timeseries inputs must have same length")
    if n_boot <= 1:
        raise ValueError("n_boot must be > 1")
    output = np.empty(n_boot)
    indices = np.arange(real.size)
    # bootstrap loop
    for i in range(n_boot):
        choose = np.random.choice(indices, indices.size, replace=True)
        full_r2 = np.corrcoef(real[choose], predicted[choose])[0, 1]**2
        rem_r2 = np.corrcoef(real[choose], remainder[choose])[0, 1]**2
        output[i] = 1 - rem_r2 / full_r2
    return output


def bootstrap_binned_average(x: np.ndarray, weights: np.ndarray, bins: np.ndarray, n_boot: int) -> np.ndarray:
    """
    Returns bootstrap samples for a weighted histogram (i.e. average of a quantity based on bins of another quantity)
    :param x: The quantity to bin by
    :param weights: The quantity to average
    :param bins: Array of bin boundaries
    :param n_boot: The number of bootstrap samples to generate
    :return: n_boot x (bins.size-1) matrix of binned average bootstrap samples
    """
    if x.ndim > 1 or weights.ndim > 1:
        raise ValueError("Only 1D vectors are supported for x and weights")
    if x.size != weights.size:
        raise ValueError("Each x sample must have an associated weight")
    output = np.empty((n_boot, bins.size - 1))
    indices = np.arange(x.size)
    for i in range(n_boot):
        choose = np.random.choice(indices, indices.size, replace=True)
        x_bs = x[choose]
        w_bs = weights[choose]
        weighted_count = np.histogram(x_bs, bins, weights=w_bs)[0].astype(float)
        count = np.histogram(x_bs, bins)[0].astype(float)
        output[i, :] = weighted_count / count
    return output


def jacknife_entropy(data: np.ndarray, nbins: int) -> float:
    """
    Computes the jacknife estimate of the entropy of data according to (Zahl, Ecology, 1977)
    :param data: nsamples x nfeatures sized data matrix
    :param nbins: The number of histogram bins to use along each dimension
    :return The jacknife estimate of the entropy in bits
    """
    hist = np.histogramdd(data, nbins)[0].ravel()
    ent_full = entropy(hist, base=2)
    # jacknife
    jk_sum = 0
    jk_n = 0
    for i in range(hist.size):
        if hist[i] > 0:
            jk_hist = hist.copy()
            jk_hist[i] -= 1
            # for each element in this bin we get exactly one jack-nife estimate
            jk_sum = jk_sum + hist[i] * entropy(jk_hist, base=2)
            jk_n += hist[i]
    return hist.sum() * ent_full - (hist.sum() - 1)*jk_sum/jk_n


def mutual_information(in1: np.ndarray, in2: np.ndarray, nbins: int) -> float:
    """
    Computes the mutual information in bits between in1 and in2
    :param in1: nsamples x nfeatures sized data matrix
    :param in2: nsamples x nfeatures sized data matrix
    :param nbins: The number of histogram bins to use along each dimension
    :return: The mutual informatio between in1 and in2 in bits
    """
    if in1.ndim == 1:
        in1 = in1[:, None]
    if in2.ndim == 1:
        in2 = in2[:, None]
    if in1.shape[0] != in2.shape[0]:
        raise ValueError("in1 and in2 must have the same number of samples")
    in1_entropy = jacknife_entropy(in1, nbins)
    in2_entropy = jacknife_entropy(in2, nbins)
    joint_ent = jacknife_entropy(np.hstack((in1, in2)), nbins)
    return in1_entropy + in2_entropy - joint_ent


def safe_standardize(x: np.ndarray, axis: Optional[int] = None, epsilon=1e-9) -> np.ndarray:
    """
    Standardizes an array to 0 mean and unit standard deviation avoiding division by 0
    :param x: The array to standardize
    :param axis: The axis along which standardization should be performmed
    :param epsilon: Small constant to add to standard deviation to avoid divide by 0 if sd(x)=0
    :return: The standardized array of same dimension as x
    """
    if x.ndim == 1 or axis is None:
        y = x - np.mean(x)
        y /= (np.std(y) + epsilon)
    else:
        y = x - np.mean(x, axis=axis, keepdims=True)
        y /= (np.std(y, axis=axis, keepdims=True) + epsilon)
    return y


def barcode_cluster(x: np.ndarray, threshold: Union[float, np.ndarray]) -> np.ndarray:
    """
    For n_samples by n_features input matrix assigns a cluster to each member based on "barcoding"
    where all above-threshold features are set to contribute to a sample
    :param x: n_samples x n_features input
    :param threshold: The trehshold(s) above which (all) features contribute - either scalar or n_features vector
    :return: n_samples long vector of cluster numbers. Ordered as if contributions were binary digits with index 0
        of x having highest significance (no contribution would be first, all contributing last)
    """

    def split(m, row_ix=None, index=0):
        if row_ix is None:
            row_ix = np.arange(m.shape[0])
        above = m[m[:, index] > 0]
        rix_above = row_ix[m[:, index] > 0]
        below = m[m[:, index] <= 0]
        rix_below = row_ix[m[:, index] <= 0]
        if above.size == 0 and index == m.shape[1]-1:
            return [rix_below]
        if below.size == 0 and index == m.shape[1]-1:
            return [rix_above]
        if index == m.shape[1]-1:
            return [rix_above, rix_below]
        if above.size == 0:
            return split(below, rix_below, index+1)
        if below.size == 0:
            return split(above, rix_above, index+1)
        return split(above, rix_above, index+1) + split(below, rix_below, index+1)

    if not np.isscalar(threshold):
        if threshold.size != x.shape[1]:
            raise ValueError("Threshold either has to be a scalar or a vector with n_features element")
        threshold = threshold.ravel()[None, :]
    xt = x > threshold
    clustered_indices = split(xt)
    cluster_numbers = np.full(x.shape[0], np.nan)
    for i, clust in enumerate(reversed(clustered_indices)):
        cluster_numbers[clust] = i
    return cluster_numbers


def create_coordinate_grid_points(x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray]) -> np.ndarray:
    """
    Creates linearized grid of all coordinate pairs
    :param x: Vector of x coordinates in the grid
    :param y: Vector of y coordinates in the grid
    :param z: None or vector of z coordinates in the grid
    :return: n x 2 or n x 3 matrix of all points in the grid
    """
    if z is None:
        xv, yv = np.meshgrid(x, y)
        xv = xv.reshape((np.prod(xv.shape),))
        yv = yv.reshape((np.prod(yv.shape),))
        return np.hstack((xv[:, None], yv[:, None]))
    else:
        xv, yv, zv = np.meshgrid(x, y, z)
        xv = xv.reshape((np.prod(xv.shape),))
        yv = yv.reshape((np.prod(yv.shape),))
        zv = zv.reshape((np.prod(zv.shape),))
        return np.hstack((xv[:, None], yv[:, None], zv[:, None]))


@njit
def rearrange_hessian(hessian: np.ndarray, npreds: int, inp_length: int) -> np.ndarray:
    """
    Re-arranges contents of our hessian matrices so that consecutive rows/columns are grouped by predictor
    instead of by time
    :param hessian: The hessian
    :param npreds: The number of predictors
    :param inp_length: The timelength of each regressor input
    :return: The re-arranged hessian
    """
    hessian_r = np.empty_like(hessian, dtype=np.float32)
    for row in range(hessian.shape[0]):
        regnum = row % npreds
        time = row // npreds
        row_ix = regnum * inp_length + time
        for col in range(hessian.shape[1]):
            regnum = col % npreds
            time = col // npreds
            col_ix = regnum * inp_length + time
            hessian_r[row_ix, col_ix] = hessian[row, col]
    return hessian_r


def simulate_response(act_predictor: model.ActivityPredictor, predictors: np.ndarray) -> np.ndarray:
    """
    Simulate the predicted response of a neuron to an arbitrary input
    :param act_predictor: The model used to predict the response
    :param predictors: n_time x m_predictors matrix of predictor inputs
    :return: n_time - history_length + 1 long vector of predicted neural responses
    """
    history = act_predictor.input_length
    pred = [act_predictor.get_output(predictors[None, t - history + 1:t + 1, :]) for t in
            range(history - 1, predictors.shape[0])]
    return np.hstack(pred)


def create_time_shifted_predictor_matrix(raw_predictors: np.ndarray, input_length: int) -> np.ndarray:
    """
    To form comparison linear regression models converts predictor inputs across time into an augmented matrix
    which contains time-shifted versions of all predictors and ortho-normalizes all augmented predictors
    :param raw_predictors: All predictors across time n_timepoints x m_predictors
    :param input_length: Same as model input length, uses to determine the required time-shift
    :return: Augmented and orthonormalized predictor matrix
    """
    n = raw_predictors.shape[0]
    predictors = np.full((n - input_length + 1, raw_predictors.shape[1] * input_length), np.nan)
    for ix_p in range(raw_predictors.shape[1]):
        for time_shift in range(input_length):
            predictors_ix = ix_p * input_length + time_shift
            predictors[:, predictors_ix] = raw_predictors[time_shift:n - input_length + time_shift + 1, ix_p]
    # gram-schmidt orthogonalize
    predictors = modified_gram_schmidt(predictors)
    return predictors


def modified_gram_schmidt(col_mat: np.ndarray) -> np.ndarray:
    """
    Performs orthogonalization of col_mat such that in case of linear dependence, linearly
    dependent columns will be set to all 0
    :param col_mat: mxn matrix with columns containing features
    :return: mxn matrix with orthogonalized columns
    """
    # initialize with copy
    v = col_mat.copy()
    # iterate through all columns
    for j in range(v.shape[1]):
        # if the current column is linearly dependent to previous columns
        # its values will be close to 0 - we set to 0 exactly and move on
        if np.allclose(v[:, j], 0):
            v[:, j] = np.zeros(v.shape[0])
            continue
        n = np.linalg.norm(v[:, j])
        q = v[:, j] / n  # this is the unit vector we will  project out all *subsequent* columns one-by-one
        for k in range(j+1, v.shape[1]):
            v[:, k] = v[:, k] - (q@v[:, k])*q
    # set vector lengths to unit norm (in a safe manner avoiding div-0 for 0 vectors)
    norms = np.linalg.norm(v, axis=0, keepdims=True)
    norms[norms < 1e-9] = 1e-9
    v /= norms
    return v


class NonlinClassifier:
    """
    Implements the logistic regression model on standardized data we use to predict whether a neuron is
    nonlinear or not
    """
    def __init__(self, curve_mean: float, curve_std: float, nlc_mean: float, nlc_std: float, lrm_curve_weight: float,
                 lrm_nlc_weight: float, lrm_intercept: float):
        """
        Creates a new NonlinClassifier
        :param curve_mean: The mean of the curvature training data
        :param curve_std: The standard deviation of the curvature training data
        :param nlc_mean: The mean of the nlc training data
        :param nlc_std: The standard deviation of the nlc training data
        :param lrm_curve_weight: The weight coefficient of the curvature term in the model
        :param lrm_nlc_weight: The weight coefficient of the nlc term in the model
        :param lrm_intercept: The model intercept
        """
        self._curve_mean = curve_mean
        self._curve_std = curve_std
        self._nlc_mean = nlc_mean
        self._nlc_std = nlc_std
        self._lrm_curve_weight = lrm_curve_weight
        self._lrm_nlc_weight = lrm_nlc_weight
        self._lrm_intercept = lrm_intercept

    def nonlin_probability(self, curve_metrics: np.ndarray, nlc_metrics: np.ndarray) -> np.ndarray:
        """
        Computes the probability that given samples are non-linear
        :param curve_metrics: n element vector of raw curvature metrics
        :param nlc_metrics: n element vector of raw nlc metrics
        :return: n element vector of nonlinearity probabilities
        """
        c = (curve_metrics-self._curve_mean) / self._curve_std
        n = (nlc_metrics-self._nlc_mean) / self._nlc_std
        log_probs = self._lrm_intercept + self._lrm_curve_weight*c + self._lrm_nlc_weight*n
        return 1 / (1 + np.exp(-log_probs))

    def is_nonlinear(self, threshold: float, curve_metrics: np.ndarray, nlc_metrics: np.ndarray) -> np.ndarray:
        """
        Classifies units into linear vs. nonlinear units
        :param threshold: The inclusive classification threshold 0<threshold<1
        :param curve_metrics: n element vector of raw curvature metrics
        :param nlc_metrics: n element vector of raw nlc metrics
        :return: n element bool vector of whether units are non-linear or not
        """
        probs = self.nonlin_probability(curve_metrics, nlc_metrics)
        return probs >= threshold

    @staticmethod
    def get_standard_model():
        """
        Returns classification model pre-filled with the standard parameters found in cnn_nonlin_test.py
        """
        nlc_mean = 1.3485983662263608
        nlc_std = 1.3843052205936655
        crv_mean = 0.02291691061897682
        crv_std = 0.03705165703426925
        lrm_crv_weight = 6.312495358261913
        lrm_nlc_weight = 7.246720236165673
        lrm_intercept = 4.27271246
        return NonlinClassifier(crv_mean, crv_std, nlc_mean, nlc_std, lrm_crv_weight, lrm_nlc_weight, lrm_intercept)


class SpatialQuery:
    """
    Class to query spatial properties of neuron centroids
    """
    def __init__(self, all_neuron_centroids: np.ndarray, selector: np.ndarray):
        """
        Creates a new SpatialQuery class
        :param all_neuron_centroids: n x dim  array of centroids across all neurons, background and class of interest
        :param selector: Boolean vector of size n indicating which centroids belong to class of interest
        """
        self._neuron_centroids = all_neuron_centroids.copy()
        self._selector = selector.copy()
        self._bootsel = selector.copy()
        self._ndim = all_neuron_centroids.shape[1]
        # Cashes of trees to avoid re-construction
        self._select_tree: Optional[KDTree] = None
        self._boot_trees: List[KDTree] = []

    @staticmethod
    def _make_tree(centroids) -> KDTree:
        """
        Helper function to create a KDTree of non-nan centroids
        """
        val = np.sum(np.isnan(centroids), 1) < 1
        return KDTree(centroids[val])

    def _check_dim(self, a: np.ndarray):
        if self._ndim != a.shape[1]:
            raise ValueError(f"Dimensionality (size of axis 1) of all point inputs must match centroid "
                             f"dimension of {self._ndim}")

    def _get_boot_tree(self, index: int) -> KDTree:
        """
        If necessary creates and then returns a KDTree at a given bootstrap index which has the selector
        coordinates randomly shuffled with respect to all passed-in centroids
        """
        # Generate all bootstrap samples up to index
        while len(self._boot_trees) <= index:
            np.random.shuffle(self._bootsel)
            self._boot_trees.append(self._make_tree(self._neuron_centroids[self._bootsel]))
        return self._boot_trees[index]

    def spatial_density(self, query_points: np.ndarray, sd_radius: float) -> np.ndarray:
        """
        For a set of points computes the kernel density of neuron centroids within a certain distance from each point
        :param query_points: n_q_points x n_dim array of locations for which to count the objects
        :param sd_radius: The ball/circle standard deviation for the kernel density
        :return: n_q_points long vector of kernel density estimates
        """
        self._check_dim(query_points)
        return self.select_tree.kernel_density(query_points, sd_radius)

    def enrichment_by_distance(self, query_points: np.ndarray, k_neighbors: int, n_boot: int) -> np.ndarray:
        self._check_dim(query_points)
        dist_to_selected = self.select_tree.query(query_points, k_neighbors)[0]
        if k_neighbors > 1:
            dist_to_selected = np.mean(dist_to_selected, 1)
        boot_dists = []
        for i in range(n_boot):
            dist_to_boot = self._get_boot_tree(i).query(query_points, k_neighbors)[0]
            if k_neighbors > 1:
                dist_to_boot = np.mean(dist_to_boot, 1)
            boot_dists.append(dist_to_boot)
        if query_points.shape[0] > 1:
            boot_dists = np.vstack(boot_dists)
            mbd = np.mean(boot_dists, 0)
        else:
            mbd = np.mean(boot_dists)
        return mbd / (dist_to_selected+mbd)

    @property
    def select_tree(self):
        if self._select_tree is None:
            self._select_tree = self._make_tree(self._neuron_centroids[self._selector])
        return self._select_tree


class Data:
    def __init__(self, input_steps, regressors: list, ca_responses: np.ndarray, tsteps_for_train=-1):
        """
        Creates a new Data class
        :param input_steps: The number of regressor timesteps into the past to use to model the response
        :param regressors: List of regressors. Vectors are shared for all ca_responses, while matrices must have
            same shape as ca_responses
        :param ca_responses: n_responses x m_timesteps matrix of cell calcium responses
        :param tsteps_for_train: If negative use all samples for training if positive use first m samples only
        """
        self.data_len = ca_responses.shape[1]
        self.n_responses = ca_responses.shape[0]
        self.regressors = []
        for i, reg in enumerate(regressors):
            if reg.ndim > 2:
                raise ValueError(f"Regressor {i} has more than 2 dimensions")
            elif reg.ndim == 2:
                if reg.shape[0] != 1 and reg.shape[0] != self.n_responses:
                    raise ValueError(f"Regressor {i} is matrix but does not have same amount of samples "
                                     f"as ca_responses")
                if reg.shape[1] != self.data_len:
                    raise ValueError(f"Regressor {i} needs to have same amount of timesteps as ca_responses")
            else:
                if reg.size != self.data_len:
                    raise ValueError(f"Regressor {i} needs to have same amount of timesteps as ca_responses")
                reg = reg[None, :]  # augment shape
            self.regressors.append(reg)
        self.ca_responses = ca_responses
        self.input_steps = input_steps
        if tsteps_for_train > 0:
            self.tsteps_for_train = tsteps_for_train
        elif tsteps_for_train == 0:
            raise ValueError("tsteps_for_train has to be either negative or larger 0")
        else:
            self.tsteps_for_train = ca_responses.shape[1]

    def training_data(self, sample_ix: int, batch_size=32):
        """
        Creates training data for the indicated calcium response sample (cell)
        :param sample_ix: The index of the cell
        :param batch_size: The training batch size to use
        :return: Tensorflow dataset that can be used for training with randomization
        """
        out_data = self.ca_responses[sample_ix, self.input_steps-1:self.tsteps_for_train].copy()
        in_data = np.full((out_data.size, self.input_steps, len(self.regressors)), np.nan).astype(np.float32)
        for i, reg in enumerate(self.regressors):
            if reg.shape[0] == 1:
                this_reg = reg
            else:
                this_reg = reg[sample_ix, :][None, :]
            for t in range(self.input_steps-1, out_data.size+self.input_steps-1):
                in_data[t-self.input_steps+1, :, i] = this_reg[0, t-self.input_steps+1:t+1]
        train_ds = tf.data.Dataset.from_tensor_slices((in_data, out_data)).\
            shuffle(in_data.shape[0]).batch(batch_size, drop_remainder=True)
        return train_ds.prefetch(tf.data.AUTOTUNE)

    def test_data(self, sample_ix: int, batch_size=32):
        """
        Creates test data for the indicated calcium response sample (cell)
        :param sample_ix: The index of the cell
        :param batch_size: The training batch size to use
        :return: Tensorflow dataset that can be used for testing
        """
        if self.tsteps_for_train == self.ca_responses.shape[1]:
            raise ValueError("All data is training data")
        out_data = self.ca_responses[sample_ix, self.tsteps_for_train+self.input_steps-1:].copy()
        in_data = np.full((out_data.size, self.input_steps, len(self.regressors)), np.nan).astype(np.float32)
        for i, reg in enumerate(self.regressors):
            if reg.shape[0] == 1:
                this_reg = reg
            else:
                this_reg = reg[sample_ix, :][None, :]
            for t in range(self.input_steps-1, out_data.size+self.input_steps-1):
                t_t = t + self.tsteps_for_train
                in_data[t-self.input_steps+1, :, i] = this_reg[0, t_t-self.input_steps+1:t_t+1]
        test_ds = tf.data.Dataset.from_tensor_slices((in_data, out_data)).batch(batch_size, drop_remainder=False)
        return test_ds.prefetch(tf.data.AUTOTUNE)

    def regressor_matrix(self, sample_ix: int) -> np.ndarray:
        """
        For a given sample returns regressor matrix
        :param sample_ix: The index of the cell
        :return: n_timesteps x m_regressors matrix of regressors for the given cell
        """
        reg_data = np.full((self.ca_responses.shape[1], len(self.regressors)), np.nan).astype(np.float32)
        for i, reg in enumerate(self.regressors):
            if reg.shape[0] == 1:
                this_reg = reg
            else:
                this_reg = reg[sample_ix, :]
            reg_data[:, i] = this_reg.ravel().copy()
        return reg_data

    def predict_response(self, sample_ix: int, act_predictor: model.ActivityPredictor):
        """
        Obtains the predicted response for a given cell
        :param sample_ix: The index of the cell
        :param act_predictor: The model to perform the prediction
        :return:
            [0]: n_timesteps-input_steps+1 sized vector of response prediction
            [1]: Corresponding timesteps in the original calcium response
        """
        if act_predictor.input_length != self.input_steps:
            raise ValueError("Input length of activity prediction model and data class mismatch")
        regressors = self.regressor_matrix(sample_ix)
        pred = simulate_response(act_predictor, regressors)
        return pred, self.ca_responses[sample_ix, self.input_steps-1:]

    def subset(self, sample_indices: List[int]) -> Data:
        """
        Returns a subset of the data, copying  responses and predictors for the indicated cell indices
        :param sample_indices: List of cell indices which should be contained in the new data object
        :return: A data object with only the indicated subset of cells and associated predictors
        """
        new_regs = []
        for r in self.regressors:
            if r.shape[0] == 1:
                new_regs.append(r.copy())
            else:
                new_regs.append(r[sample_indices, :].copy())
        new_ca_responses = self.ca_responses[sample_indices, :].copy()
        return Data(self.input_steps, new_regs, new_ca_responses, self.tsteps_for_train)

    def save(self, filename: str, overwrite=False) -> None:
        """
        Serializes the data instance to an hdf5 file
        :param filename: The path and name of the hdf5 file to save to
        :param overwrite: If set to true and file already exists it will be overwritten if False and exists will fail
        """
        with h5py.File(filename, mode='w' if overwrite else 'x') as dfile:
            self.save_direct(dfile, overwrite)

    def save_direct(self, file: Union[h5py.File, h5py.Group], overwrite=False) -> None:
        """
        Serializes the data instance to an hdf5 file object or group in a file
        :param file: The hdf5 file object
        :param overwrite: If set to true and contents already exists will be overwritten if False and exists will fail
        """
        create_overwrite(file, "input_steps", self.input_steps, overwrite)
        create_overwrite(file, "tsteps_for_train", self.tsteps_for_train, overwrite)
        create_overwrite(file, "ca_responses", self.ca_responses, overwrite)
        create_overwrite(file, "n_regressors", len(self.regressors), overwrite)  # saved  for purposes of easy loading
        if "regressors" not in file:
            r_group = file.create_group("regressors")
        else:
            r_group = file["regressors"]
        for i, r in enumerate(self.regressors):
            create_overwrite(r_group, str(i), r, overwrite)

    @staticmethod
    def load(filename: str) -> Data:
        """
        Loads a stored data instance from an hdf5 file
        :param filename: The path and name of the hdf5 file containing the stored data instance
        :return: A Data object with the contents loaded from file
        """
        with h5py.File(filename, mode='r') as dfile:
            return Data.load_direct(dfile)

    @staticmethod
    def load_direct(file: Union[h5py.File, h5py.Group]) -> Data:
        """
        Loads a stored data instance directly from an hdf5 file object or group
        :param file: The hdf5 file object
        :return: A Data object with the contents loaded from file
        """
        input_steps = file["input_steps"][()]
        tsteps_for_train = file["tsteps_for_train"][()]
        ca_responses = file["ca_responses"][()]
        n_regressors = file["n_regressors"][()]
        r_group = file["regressors"]
        regressors = []
        for i in range(n_regressors):
            regressors.append(r_group[str(i)][()])
        return Data(input_steps, regressors, ca_responses, tsteps_for_train)


if __name__ == "__main__":
    print("Module with data preparation classes and functions")
