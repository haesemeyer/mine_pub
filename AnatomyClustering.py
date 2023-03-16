"""
Module written by Kaarthik A. Balakrishnan for anatomical clustering of coordinates
"""


import numpy as np
import matplotlib.pyplot as pl
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import cc3d
import seaborn as sns
from os import path
import matplotlib as mpl


def Cluster_Neurons(all_centroids_um, radii=None, min_count=10):
    """
    Perform anatomical clustering
    :param all_centroids_um: Coordinates to be clustered in um
    :param radii: The radius to merge coordinates into clusters in um
    :param min_count: The minimum number of coordinates in a cluster for them to receive a plot alpha > 0
    :return:
    """
    # All positions of neurons in 3D space; Convert to integer coordinates; resolution is in microns
    if radii is None:
        radii = [5]
    integer_coords = np.round(all_centroids_um).astype(np.int64)

    Z_pos = np.zeros((1200, 1200, 300))
    for i in range(len(integer_coords)):
        x = integer_coords[i, 0]
        y = integer_coords[i, 1]
        z = integer_coords[i, 2]
        if(x<-1000):
            continue
        Z_pos[x, y, z] = 1

    labels_list = {}
    number_cluster = {}
    alpha_labels = {}
    labeled_coordinates = {}

    for radius in radii:

        # Generating the kernel for ball
        print("Now using radius:",radius)
        z = np.zeros((200, 200, 200))
        test_x = 100
        test_y = 100
        test_z = 100
        z[test_x, test_y, test_z] = 1
        gz = gaussian_filter(z, sigma=20, truncate=2)
        thresh = np.mean([gz[test_x - radius, test_y, test_z], gz[test_x + radius, test_y, test_z],
                          gz[test_x, test_y - radius, test_z], gz[test_x, test_y + radius, test_z],
                          gz[test_x, test_y, test_z - radius], gz[test_x, test_y, test_z + radius]])
        gz[gz < thresh] = 0
        kern = gz[test_x - radius:test_x + radius + 1, test_y - radius:test_y + radius + 1,
               test_z - radius:test_z + radius + 1]
        kern[kern > 0] = 1

        # Convolve the neuron positions with ball to generate region for each neuron
        GZ = ndimage.convolve(Z_pos, kern)
        GZ[GZ > 0] = 1

        # Clustering neurons with the given radius
        labels_in = GZ
        labels_in[labels_in > 0] = 1
        labels_out = cc3d.connected_components(labels_in, connectivity=18)  # 18-connected
        labels_list[radius] = labels_out
        unique_labels = np.unique(labels_out)
        number_cluster[radius] = np.zeros(unique_labels.size)
        alpha_labels[radius] = {}
        print("Number of clusters:", len(unique_labels))
        print(f"Number of neurons:{len(all_centroids_um)}")
        neurons_per_cluster = np.multiply(labels_out, Z_pos)
        assert np.allclose(neurons_per_cluster, neurons_per_cluster.astype(int))
        npr = neurons_per_cluster.ravel().copy()
        # remove the background "cluster"
        npr = npr[npr != 0].astype(int)
        for element in npr:
            # iterating over all elements is faster than iterating over the (fewer) clusters
            # and repeatedly counting across the entire neurons_per_cluster array
            number_cluster[radius][element] += 1

        max_cluster = np.max(number_cluster[radius])
        for i in unique_labels[1:]:
            count = number_cluster[radius][i]
            if count < min_count:
                alpha_labels[radius][i] = 0
            else:
                alpha_labels[radius][i] = count / max_cluster * 0.5 + 0.5

        labeled_coordinates[radius] = np.full((len(all_centroids_um), 5), np.nan)
        for i in range(len(all_centroids_um)):
            if np.any(np.isnan(all_centroids_um[i])):
                continue
            temp_label=labels_list[radius][integer_coords[i, 0], integer_coords[i, 1], integer_coords[i, 2]]
            alpha_val = alpha_labels[radius][temp_label]
            labeled_coordinates[radius][i, 0:3] = all_centroids_um[i, :]
            labeled_coordinates[radius][i, 3] = alpha_val
            labeled_coordinates[radius][i, 4] = temp_label

    return Z_pos, labels_list, number_cluster, alpha_labels, labeled_coordinates


def AnatomyPlotting(labeled_coordinates, ax=None, SideView=False, color=None):
    """Take the set of labeled coordinates: the positions in 3d along with an alpha value for each position and the
     label and plot it in 3D space. ax must be a subplot with a 3d projection.
    """
    if ax is None:
        fig, ax = pl.subplots()

    if color is None:
        color = (0, 0, 0)

    colors = [(color[0], color[1], color[2], alpha) for alpha in labeled_coordinates[:, 3]]

    if SideView:
        ax.scatter(labeled_coordinates[:, 1], labeled_coordinates[:, 2], s=1, c=colors)
        ax.axis('equal')
    else:
        ax.scatter(labeled_coordinates[:, 0], labeled_coordinates[:, 1], s=1, c=colors)
        ax.axis('equal')


def PlottingbyClusters(labels_list, number_cluster, cluster, ax):
    colors = ['k']
    for i in number_cluster:
        labels_rad = labels_list[i]
        for j in number_cluster[i]:
            plotted_pixels = np.nonzero(labels_rad == j)
            region_vals = np.zeros(labels_rad.shape)
            region_vals[plotted_pixels] = 1
            ax.voxels(region_vals, color=colors[0], alpha=cluster[i][j], edgecolor=None)
        ax.set_xlabel('Medial Lateral axis')
        ax.set_ylabel('Caudal-Rostral axis')
        ax.set_zlabel('Dorsal-ventral axis')


def plot_anatomy(labeled_coords: np.ndarray, bg_centroids: np.ndarray, plot_name: str, plot_dir: str,
                 plot_color: str) -> None:
    """
    Make anatomy scatter plots
    :param labeled_coords: The cluster labeled coordinates
    :param bg_centroids: Background centroids to plot for brain outline
    :param plot_name: Plot name for saving
    :param plot_dir: Directory for saving plots
    :param plot_color: Color of labeled centroids
    """
    fig, ax = pl.subplots()
    ax.scatter(bg_centroids[:, 0], bg_centroids[:, 1], s=1, alpha=0.05, color='k')
    AnatomyPlotting(labeled_coords, ax, False, mpl.colors.to_rgb(plot_color))
    sns.despine(fig, ax)
    fig.savefig(path.join(plot_dir, f"{plot_name}_Anatomy_Top.pdf"))

    fig, ax = pl.subplots()
    ax.scatter(bg_centroids[:, 1], bg_centroids[:, 2], s=1, alpha=0.05, color='k')
    AnatomyPlotting(labeled_coords, ax, True, mpl.colors.to_rgb(plot_color))
    sns.despine(fig, ax)
    fig.savefig(path.join(plot_dir, f"{plot_name}_Anatomy_Side.pdf"))
