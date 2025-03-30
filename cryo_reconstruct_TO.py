import numpy as np

from cryo_TO_group_elements import cryo_TO_group_elements

from aspire.image import Image
from aspire.utils.rotation import Rotation
from aspire.source import ArrayImageSource
from aspire.basis import FFBBasis3D
from aspire.reconstruction import WeightedVolumesEstimator

def cryo_reconstruct_TO(symmetry,projs,rots,n_r,n_theta,max_shift,shift_step, cg_max_iterations):
    """

    :param symmetry:    Symmetry type: 'T' or 'O'.
    :param projs:       Numpy array of projection images (n_images x img_size x img_size).
    :param rots:        Array of estimated rotation matrices for each projection image
    :param n_theta:     Angular resolution for common lines detection.
    :param n_r:         Radial resolution.
    :param max_shift:   Maximal 1d shift (in pixels)
    :param shift_step:  Resolution of shift estimation in pixels.

    :return:            Estimated reconstruction of the volume (aspire Volume)
    """
    imgs = Image(projs)
    rots = Rotation(rots)

    src = ArrayImageSource(imgs, angles=rots.angles, symmetry_group=symmetry)
    basis = FFBBasis3D(src.L, dtype=src.dtype)
    weights = np.ones((src.n, 1)) / np.sqrt(src.n * len(cryo_TO_group_elements(symmetry)[0]))

    estimator = WeightedVolumesEstimator(weights, src, basis, preconditioner="none", maxiter=cg_max_iterations)
    estimated_volume = estimator.estimate()

    return estimated_volume