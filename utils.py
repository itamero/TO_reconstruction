import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from cryo_TO_group_elements import cryo_TO_group_elements
from warnings import catch_warnings, filterwarnings
from aspire.utils.rotation import Rotation


def show(imgs, columns=5, figsize=(20, 10), colorbar=False, Title=None):
    """
    Plotting Utility Function.

    :param columns: Number of columns in a row of plots.
    :param figsize: Figure size in inches, consult `matplotlib.figure`.
    :param colorbar: Optionally plot colorbar to show scale.
        Defaults to True. Accepts `bool` or `dictionary`,
        where the dictionary is passed to `matplotlib.pyplot.colorbar`.
    """

    if imgs.stack_ndim > 1:
        raise NotImplementedError("`show` is currently limited to 1D image stacks.")

    # We never need more columns than images.
    columns = min(columns, imgs.n_images)
    rows = (imgs.n_images + columns - 1) // columns  # ceiling divide.

    # Create an empty colorbar options dictionary as needed.
    colorbar_opts = colorbar if isinstance(colorbar, dict) else dict()

    # Create a context manager for altering warnings
    with catch_warnings():
        # Filter off specific warning.
        # sphinx-gallery overrides to `agg` backend, but doesn't handle warning.
        filterwarnings(
            "ignore",
            category=UserWarning,
            message="Matplotlib is currently using agg, which is a"
                    " non-GUI backend, so cannot show the figure.",
        )

        plt.figure(figsize=figsize)
        for i, im in enumerate(imgs.asnumpy()):
            plt.subplot(rows, columns, i + 1)
            plt.imshow(im, cmap="gray")
            if colorbar:
                plt.colorbar(**colorbar_opts)
        plt.suptitle(Title, fontsize=32)
        plt.show()


def mean_angular_distance_sym(rots_1, rots_2, sym):
    """"
    Find the mean angular distance between two sets of rotation matrices accounting for symmetry
    """
    ang_dists = np.zeros(len(rots_1))
    gR = cryo_TO_group_elements(sym)[0]
    for ind in range(len(rots_1)):
        R_sym = np.array([g_R @ rots_1[ind] for g_R in gR])
        dist = np.min(Rotation.angle_dist(rots_2[ind], R_sym))
        ang_dists[ind] = dist
    return np.mean(ang_dists) * 180 / np.pi # Convert radians to degrees
