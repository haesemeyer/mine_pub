"""
File to transform the processed data from Musall et al., 2019 into plot panels
"""


import argparse
import os
from os import path
from typing import Any
import matplotlib as mpl
import matplotlib.pyplot as pl
import seaborn as sns
import h5py
import numpy as np
import pandas as pd

import utilities


class CheckArgs(argparse.Action):
    """
    Check our command line arguments for validity
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values: Any, option_string=None):
        if self.dest == 'file':
            if not path.exists(values):
                raise argparse.ArgumentError(self, f"Specified file {values} does not exist")
            if not path.isfile(values):
                raise argparse.ArgumentError(self, f"The destination {values} is not a file")
            setattr(namespace, self.dest, values)
        else:
            raise Exception("Parser was asked to check unknown argument")


class TaylorMap:
    """
    Class to map taylor decomposition information into pixel space
    """
    def __init__(self, session_group: h5py.Group):
        """
        Create a new TaylorMap object
        :param session_group: The session for which to do the mapping
        """
        # load nonlinearity probabilities to use them to indicate to us which fits were above threshold so that
        # Taylor metrics got computed
        nlp = session_group["output_data"]["nl_probs"][()]
        valid = np.isfinite(nlp)
        spatial = session_group["input_data"]["spatial"][()]
        self.image_dims = (spatial.shape[1], spatial.shape[2])
        # we weight all component related data by how much the model can actually explain
        this_test = session_group["output_data"]["correlations_test"][()][valid, None]
        self.g_o_f = this_test ** 2
        self.g_o_f[this_test < 0] = 0  # if test correlation was < 0 discard component
        self.u = np.reshape(spatial, (spatial.shape[0], spatial.shape[1] * spatial.shape[2]))  # reshape to real U
        self.u = self.u[valid]  # restrict to decomposed components
        self.tay_real = session_group["output_data"]["taylor_true_change"][
            ()]  # Real ANN delta-output for each component
        self.tay_real *= self.g_o_f
        self.tay_real_pix = self.tay_real.T @ self.u  # Real ANN delta-output in pixel-space
        self.tay_full = session_group["output_data"]["taylor_full_prediction"][
            ()]  # Full Taylor prediction for each component
        self.tay_full *= self.g_o_f
        tay_full_pix = self.tay_full.T @ self.u  # Full Taylor prediction of ANN delta-output in pixel space
        self.taylor_by_predictor = session_group["output_data"]["taylor_by_predictor"][
            ()]  # Taylor term for each predictor
        # compute the R2 value of the full Taylor expansion in each pixel
        self.full_r2_pix = np.full(self.u.shape[1], np.nan)
        for p in range(self.full_r2_pix.size):
            self.full_r2_pix[p] = np.corrcoef(self.tay_real_pix[:, p], tay_full_pix[:, p])[0, 1] ** 2
        labels = session_group["input_data"]["predictor_labels"][()]
        self.example_labels = [lbl[0].decode() for lbl in labels]  # list of predictor labels as proper strings
        self.jacobians = session_group["output_data"]["jacobians"][()][valid]

    def create_metric_map(self, predictor_label: str) -> np.ndarray:
        """
        Create Taylor metric map for the indicated predictor
        :param predictor_label: The name of the predictor
        :return: n_pixels_h x n_pixels_w image of the taylor metric
        """
        try:
            ix_pred = self.example_labels.index(predictor_label)
        except ValueError:
            raise ValueError(f"{predictor_label} is not in the list of predictors")
        # calculate remainder for each component after removing the selected predictor - since tay_full is alread
        # scaled by g_o_f we need to do the same for the component we remove
        remainder = self.tay_full - self.taylor_by_predictor[:, :, ix_pred, ix_pred] * self.g_o_f
        # map remainder into pixel space
        remainder_pix = remainder.T @ self.u
        # for each pixel compute the average R2 of the remainder in each pixel
        remainder_r2_pix = np.full(self.u.shape[1], np.nan)
        for i in range(self.full_r2_pix.size):
            remainder_r2_pix[i] = np.corrcoef(self.tay_real_pix[:, i], remainder_pix[:, i])[0, 1] ** 2
        # compute the taylor metric in pixel space and reformat to image
        taylor_metric_map = 1 - remainder_r2_pix / self.full_r2_pix
        taylor_metric_map = taylor_metric_map.reshape(self.image_dims)
        # values < 0 are not meaningful
        taylor_metric_map[taylor_metric_map < 0] = 0
        return taylor_metric_map.T  # transpose to map into original reference space (otherwise L/R inverted)

    def create_jacobian_map(self, predictor_label: str) -> np.ndarray:
        """
        Create a per-pixel map of jacobians of the indicated predictor
        :param predictor_label: The name of the predictor
        :return: n_timepoints x n_pixels_h x n_pixels_w array of the jacobians
        """
        try:
            ix_pred = self.example_labels.index(predictor_label)
        except ValueError:
            raise ValueError(f"{predictor_label} is not in the list of predictors")
        # map jacobian into pixel space
        jpix = (self.jacobians[:, 150*ix_pred:150*(ix_pred+1)]*self.g_o_f).T @ self.u
        jpix = np.reshape(jpix, (jpix.shape[0], self.image_dims[0], self.image_dims[1]))  # reshape to image space
        # transpose image dimensions
        jp_map = np.empty((jpix.shape[0], jpix.shape[2], jpix.shape[1]))
        for r in range(jpix.shape[1]):
            for c in range(jpix.shape[2]):
                jp_map[:, c, r] = jpix[:, r, c]
        return jp_map

    def cluster_jacobian_map(self, predictor_label: str, n_clusters: int, taylor_percentile=90.0) -> [np.ndarray,
                                                                                                      np.ndarray]:
        """
        Cluster jacobians in pixel space for pixels that are in the top percentile of being related to the predictor
        :param predictor_label: The name of the predictor
        :param n_clusters: The number of clusters
        :param taylor_percentile: The threshold which pixels have to cross for their taylor metric to be included
        :return:
            [0]: n_valid_pixels long vector of cluster labels
            [1]: The included pixel-space jacobians (n_valid_pixels x n_timepoints)
        """
        jac_map = self.create_jacobian_map(predictor_label)
        jac_map = jac_map.reshape((jac_map.shape[0], jac_map.shape[1]*jac_map.shape[2]))
        tay_map = self.create_metric_map(predictor_label).ravel()
        map_valid = tay_map > np.nanpercentile(tay_map, taylor_percentile)
        jac_map = jac_map[:, map_valid].T
        clust_ids = utilities.spectral_cluster(jac_map, n_clusters, 0.8)
        return clust_ids, jac_map


def plot_session_maps(anatsession: h5py.Group, plot_dir: str, ext: str) -> TaylorMap:
    session_name = anatsession.name.replace('/', '')
    mapper = TaylorMap(anatsession)
    left_vis = mapper.create_metric_map("lVisStim")
    right_vis = mapper.create_metric_map("rVisStim")
    right_grab = mapper.create_metric_map("rGrab")
    left_grab = mapper.create_metric_map("lGrab")
    whisk = mapper.create_metric_map("whisk")

    fig = pl.figure()
    sns.heatmap(left_vis - right_vis, center=0, xticklabels=100, yticklabels=100, rasterized=True, cmap='vlag',
                vmax=0.4, vmin=0-.4)
    pl.axis('equal')
    fig.savefig(path.join(plot_dir, f"{session_name}_Map_Left_vs_Right_VisualStim.{ext}"), dpi=600)

    fig = pl.figure()
    sns.heatmap(whisk - (left_vis + right_vis) / 2, center=0, xticklabels=100, yticklabels=100, rasterized=True,
                cmap='vlag', vmax=0.4, vmin=-0.4)
    pl.axis('equal')
    fig.savefig(path.join(plot_dir, f"{session_name}_Map_Whisk_vs_VisualStim.{ext}"), dpi=600)

    fig = pl.figure()
    sns.heatmap(left_grab - right_grab, center=0, xticklabels=100, yticklabels=100, rasterized=True, cmap='vlag',
                vmax=0.4, vmin=-0.4)
    pl.axis('equal')
    fig.savefig(path.join(plot_dir, f"{session_name}_Map_Left_vs_Right_Grab.{ext}"), dpi=600)

    return mapper


def main(cl_args):
    data_file = cl_args.file
    as_png = cl_args.save_png

    ext = "png" if as_png else "pdf"

    data_folder, data_filename = path.split(data_file)
    base_name = path.splitext(data_filename)[0]

    plot_dir = path.join(data_folder, f"{base_name}_figure_panels")
    if not path.exists(plot_dir):
        os.makedirs(plot_dir)

    with h5py.File(data_file, 'r') as dfile:
        sessions = sorted(list(dfile.keys()))
        session_groups = [dfile[sess] for sess in sessions]

        # collect all train and test correlations
        all_train_corrs = np.hstack([sg["output_data"]["correlations_trained"][()] for sg in session_groups])
        session_test_corrs = [sg["output_data"]["correlations_test"][()] for sg in session_groups]
        all_test_corrs = np.hstack(session_test_corrs)

        # scatter-plot comparing training and test correlations
        fig = pl.figure()
        min_c = min([np.min(all_train_corrs), np.min(all_test_corrs)])
        max_c = max([np.max(all_train_corrs), np.max(all_test_corrs)])
        pl.scatter(all_train_corrs, all_test_corrs, alpha=0.25)
        pl.plot([min_c, max_c], [min_c, max_c], 'k--')
        pl.plot([min_c, max_c], [0.1, 0.1], 'C3--')
        pl.xlabel("Train data correlations")
        pl.ylabel("Test data correlations")
        sns.despine()
        fig.savefig(path.join(plot_dir, f"Test_vs_train_corrs_scatter.{ext}"), dpi=300)

        print(f"{np.sum(all_test_corrs >= 0.1)} out of {all_test_corrs.size} components were successfully fit.")
        per_sess_fit = np.hstack([np.sum(stc > 0.1) / stc.size for stc in session_test_corrs])
        fig = pl.figure()
        sns.barplot(y=per_sess_fit)
        sns.stripplot(y=per_sess_fit, color='k')
        pl.ylabel("Fraction of identified components")
        sns.despine()
        fig.savefig(path.join(plot_dir, f"Fraction_Identified.{ext}"), dpi=300)

        # collect all nonlinearity probabilities
        all_nl_probs = np.hstack([sg["output_data"]["nl_probs"][()] for sg in session_groups])
        # kde-plot of nonlinearity probabilities
        fig = pl.figure()
        sns.kdeplot(all_nl_probs)
        pl.xlim(0, 1)
        pl.xlabel("Nonlinearity probability")
        pl.ylabel("Density")
        sns.despine()
        fig.savefig(path.join(plot_dir, f"Nonlinearity_KDE.{ext}"), dpi=300)

        # collect all Taylor scores - NOTE: The following is only sensible if the predictor labels are
        # the same across all sessions, so we make sure of that
        label_list = [sg["input_data"]["predictor_labels"][()] for sg in session_groups]
        for s_index, sess_labels in enumerate(label_list):
            for l_index in range(22):
                if sess_labels[l_index] != label_list[0][l_index]:
                    print(f"Mismatch detected in session {s_index}")
        all_taylor_raw = np.vstack([sg["output_data"]["taylor_scores"][:, :, 0] for sg in session_groups])
        all_taylor_se = np.vstack([sg["output_data"]["taylor_scores"][:, :, 1] for sg in session_groups])
        # set signficance to 3 sigma - no correction as we apply in the massive zebrafish dataset
        sig = (all_taylor_raw - 3 * all_taylor_se) > 0
        all_taylor_scores = np.zeros_like(all_taylor_raw)
        all_taylor_scores[sig] = all_taylor_raw[sig]
        # restrict to non-interaction terms
        print(f"{np.sum(all_taylor_scores[:, 22:] > 0)} significant interaction terms found")
        all_taylor_scores = all_taylor_scores[:, :22].copy()
        # simply sort by the max-index
        all_taylor_scores_sorted = all_taylor_scores[np.argsort(np.argmax(all_taylor_scores, 1))]
        taylor_labels = [ll[0].decode() for ll in label_list[0]]
        taylor_df = pd.DataFrame(all_taylor_scores_sorted, columns=taylor_labels)
        # sort predictors by category and remove slowPupil and prevMod
        labels_sorted = ["time", "Choice", "reward", "prevReward", "prevChoice", "lGrab", "rGrab", "lLick", "rLick",
                         "piezo", "whisk", "nose", "fastPupil", "face", "body", "water", "lVisStim",
                         "rVisStim", "lAudStim", "rAudStim"]
        taylor_df = taylor_df.reindex(labels_sorted, axis=1)
        fig = pl.figure()
        sns.heatmap(taylor_df, yticklabels=250, cmap="gray_r", rasterized=True, vmax=0.5, vmin=0.0)
        fig.tight_layout()
        fig.savefig(path.join(plot_dir, f"SortedTaylorMetric_Heatmap.{ext}"), dpi=600)

        # NOTES on anatomy plots:
        # The biggest hurdle we face here is that the so-called "SVD" temporal and spatial components of the
        # Musall dataset aren't in fact orthogonal but highly dependent. If they were orthogonal we could easily
        # transfer variance of components to the variance of each pixel timeseries since in that case:
        #   var(pixel) =  var(U[0,pixel] x Vc[0]) + var(U[1,pixel] x Vc[1]) + ...
        # Similarly, we could then translate explained variance and from the ratio ev(pixel)/var(pixel) obtain an R2.
        # This would likewise mean that we could translate through our Taylor scores which are just losses in explained
        # variance.
        # However, since the components aren't independent we would need to take care of the covariances especially
        # vis-à-vis the role of positive and negative values in U.
        # However we can still transfer the variance to pixel space by computing the variances and covariances scaled by
        # the corresponding values of U:
        # var(pixel) = SUM(i) var(U[i,pixel] x Vc[i]) + SUM(i,j<i) cov(U[i,pixel] x Vc[i], U[j, pixel] x Vc[j])
        #
        # But for the explained variance this becomes tricky since it is unclear what the explained covariance is.
        # This is because 50% of explained variance of a variable can mean that it is 100% explained for half the time
        # or half explained all the time. Therefore, it is not enough information for computing the corresponding
        # covariance term. Hence, the only way to move into pixel-space is, to have the raw model and taylor reduced
        # traces. Then move these into pixel-space and calculate the metrics there.
        # So, for each term of interest we:
        # 1) Move taylor_true_change into pixel space
        # 2) Move taylor_full_prediction into pixel space
        # 3) Move (taylor_full_prediction - term_of_interest)=remainder into pixel space
        # 4) For each pixel: bootstrap_fractional_r2loss(taylor_true_change, taylor_full_prediction, remainder)
        # 5) Set non-significant pixels (as above) to 0
        # 6) Plot map

        if len(session_groups) > 1:
            example_ix = 12  # index of the example session to use
        else:
            example_ix = 0
        anatsession = session_groups[example_ix]  # anatomy plots for one example session only

        mapper = plot_session_maps(anatsession, plot_dir, ext)

        clust_lv, jac_lv = mapper.cluster_jacobian_map("lVisStim", 2)  # 2 clusters likely splits ON and OFF
        clust_rv, jac_rv = mapper.cluster_jacobian_map("rVisStim", 2)
        clust_lg, jac_lg = mapper.cluster_jacobian_map("lGrab", 2)
        clust_rg, jac_rg = mapper.cluster_jacobian_map("rGrab", 2)

        fig, axes = pl.subplots(ncols=2)
        time = (np.arange(jac_rg.shape[1]) - 74) / 30
        axes[0].plot(time, np.mean(jac_lv[clust_lv == 0], 0), label=f"Left 0 {np.sum(clust_lv == 0)}")
        axes[0].plot(time, np.mean(jac_lv[clust_lv == 1], 0), label=f"Left 1 {np.sum(clust_lv == 1)}")
        axes[0].plot(time, np.mean(jac_rv[clust_rv == 0], 0), '--', label=f"Right 0 {np.sum(clust_rv == 0)}")
        axes[0].plot(time, np.mean(jac_rv[clust_rv == 1], 0), '--', label=f"Right 1 {np.sum(clust_rv == 1)}")
        axes[0].plot([0, 0], [-0.0001, 0.0003], 'k--')
        axes[0].set_xticks([-2.5, -2.0, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0, 2.5])
        axes[0].set_xlabel("Time [s]")
        axes[0].set_ylabel("Weight [dCa/dVis]")
        axes[0].legend()
        axes[1].plot(time, np.mean(jac_lg[clust_lg == 0], 0), label=f"Left 0 {np.sum(clust_lg == 0)}")
        axes[1].plot(time, np.mean(jac_lg[clust_lg == 1], 0), label=f"Left 1 {np.sum(clust_lg == 1)}")
        axes[1].plot(time, np.mean(jac_rg[clust_rg == 0], 0), '--', label=f"Right 0 {np.sum(clust_rg == 0)}")
        axes[1].plot(time, np.mean(jac_rg[clust_rg == 1], 0), '--', label=f"Right 1 {np.sum(clust_rg == 1)}")
        axes[1].plot([0, 0], [-0.0002, 0.0002], 'k--')
        axes[1].set_xticks([-2.5, -2.0, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0, 2.5])
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Weight [dCa/dGrab]")
        axes[1].legend()
        fig.savefig(path.join(plot_dir, f"Receptive_Fields.{ext}"), dpi=300)

        # plot maps across all sessions for supplements
        plot_dir = path.join(plot_dir, "supplement")
        if not path.exists(plot_dir):
            os.makedirs(plot_dir)
        for anatsession in session_groups:
            plot_session_maps(anatsession, plot_dir, ext)


if __name__ == "__main__":
    mpl.rcParams['pdf.fonttype'] = 42

    a_parser = argparse.ArgumentParser(prog="plotMusall",
                                       description="Will plot figure panels related to MINE analysis of Musall "
                                                   "et. al 2019")
    a_parser.add_argument("-f", "--file", help="Path to the analysis hdf5 file", type=str, default="",
                          action=CheckArgs)
    a_parser.add_argument("-png", "--save_png", help="If set figures will be saved as png instead of pdf",
                          action='store_true')

    main(a_parser.parse_args())
