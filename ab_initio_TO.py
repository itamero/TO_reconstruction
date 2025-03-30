import numpy as np
import math
import os
import mrcfile
import logging

from jsync_TO import jsync_TO
from cryo_TO_create_cache import cryo_TO_create_cache, check_cache_file
from estimate_relative_rotations import estimate_relative_rotations
from estimate_rotations_synchronization import estimate_rotations_synchronization
from cryo_reconstruct_TO import cryo_reconstruct_TO
from utils import mean_angular_distance_sym

from aspire.operators import PolarFT
from aspire.image import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def cryo_abinitio_TO(sym, instack=None, outvol=None, cache_file_name=None, n_theta=360, rotation_resolution=150,
                     n_r_perc=50, viewing_direction=0.996, in_plane_rotation=5, shift_step=0.5, max_shift_perc=0,
                     cg_max_iterations=50, true_rotations=None):
    """
    Ab-initio reconstruction of a 'T' or 'O' symmetric molecule


    :param sym:                 'T' for tetrahedral symmetry, 'O' for octahedral symmetry.
    :param instack:             Name of MRC file containing the projections from which to estimate an ab-initio model.
    :param outvol:              Name (or path) of MRC file into which to save the reconstructed volume.
    :param cache_file_name:     (Optional) The pkl file name containing all candidate rotation matrices, common lines
                                indices, self common lines indices.
                                If not supplied, cache will be created.
                                Use with caution! cache_file_name needs to be in the correct format.
    :param rotation_resolution: (Optional) Number of samples per 2*pi  (see gen_rotations_grid for more details)
                                With the standard 'viewing_direction=0.996' and 'in_plane_rotation=5' the following
                                table shows the number of rotations generated per resolution:

                                Resolution | Grid Rotations  | Rotations After T/O Filtering
                                -----------|-----------------|------------------------------
                                50         | 4,484           | 1,195 (T),    724   (O)
                                75         | 15,236          | 1,541 (T),    854   (O)
                                100        | 39,365          | 1,704 (T),    887   (O)
                                150        | 129,835         | 2,084 (T),    1,068 (O)

                                Default resolution is 75

    :param in_plane_rotation:   (Optional) In-plane rotation angle threshold, used to determine proximity of rotations
                                for filtering i.e. creating "SO_G(3)"
    :param viewing_direction:   (Optional) Viewing direction angle threshold, used to determine proximity of rotations
                                for filtering i.e. creating "SO_G(3)"
    :param n_theta:             (Optional) Angular resolution for common lines detection.
                                Default 360.
    :param n_r_perc:            (Optional) Radial resolution for common line detection as a percentage of image size.
                                Default is half the width of the images.
    :param max_shift_perc:      (Optional) Maximal 1d shift (in pixels) to search between common-lines.
                                Default is 15% of image width of the images.
                                Not yet supported. Set to 0.
    :param shift_step:          (Optional) Resolution of shift estimation in pixels.
                                Default is 0.5.
    :param cg_max_iterations:   (Optional) Maximum number of iterations for CG in reconstruction.
    :param true_rotations:      (Optional) True rotations of each projection-image to be used for comparison with
                                The estimated rotations.

    :return:                    Estimated reconstruction of the volume (aspire Volume) and estimated rotations (np array)
    """

    # First argument sym must be 'T' or 'O' (tetrahedral or octahedral symmetry)
    if sym not in ['T', 'O']:
        raise NotImplementedError(
            f"Symmetry type '{sym}' was supplied. Only 'T' and 'O' are supported.")

    # Check outvol is valid
    if outvol is None:
        raise ValueError('outvol parameter not provided.')
    outvol_folder_name = os.path.dirname(outvol) # Extract folder from outvol path
    if outvol_folder_name and not os.path.isdir(outvol_folder_name): # There is a folder in the provided path but it is invalid
        raise FileNotFoundError(
            f'Folder {outvol_folder_name} does not exist. Please create it first.\n')

    ########################
    # Step 1: Cache file   #
    ########################

    if cache_file_name is None:     # Check if cache_file_name was provided, if it was - check it exists, otherwise create it
        logger.info('Cache file address not supplied.')
        expected_cache_name = f"cache_{sym}_symmetry_resolution_{rotation_resolution}_ntheta_{n_theta}.pkl"
        local_cache_exists = os.path.exists(expected_cache_name)
        if local_cache_exists:
            logger.info(f'Using existing local cache found: {expected_cache_name}\n')
            check_cache_file(expected_cache_name, sym, rotation_resolution, n_theta)
            cache_file_name = expected_cache_name
        else:
            cwd_folder = os.getcwd()
            logger.info(f'Creating cache file in folder: {cwd_folder}\n')
            cache_file_name = cryo_TO_create_cache(sym, rotation_resolution, n_theta, viewing_direction, in_plane_rotation)
    elif not os.path.isfile(cache_file_name):
        raise ValueError(
            f"Provided cache_file_name '{cache_file_name}' does not exist.")
    else:
        logger.info(f'Using provided cache file: {cache_file_name}\n')
        check_cache_file(cache_file_name, sym, rotation_resolution, n_theta)

    #############################
    # Step 2: Load projections  #
    #############################

    logger.info(f'Loading mrc image stack file: {instack}')
    with mrcfile.open(instack) as mrc:
        projs = mrc.data
    logger.info(f'Done loading mrc image stack')
    if projs.shape[1] !=  projs.shape[2]:
        raise NotImplementedError(
            f"Only square projection images are supported."
            f"Provided projection has shape ({projs.shape[1]} x {projs.shape[2]}).")
    n_images = projs.shape[0]
    img_size = projs.shape[1]

    logger.info(f'Loaded {n_images} projections of size ({projs.shape[1]} x {projs.shape[2]}).\n')
    # is the mask and gaussian filter necessary?


    ##################################################
    # Step 3: Polar Fourier transform of projections #
    ##################################################

    #projs = np.array([np.transpose(proj) for proj in projs]) # Transpose each projection to match matlab indexing

    logger.info('Computing the polar Fourier transform of projections')
    n_r = math.ceil(img_size*n_r_perc/100)
    pft = PolarFT(img_size, nrad=n_r, ntheta=n_theta)
    pf = pft.transform(Image(projs))    # pf has size (n_images) x (n_theta/2) x (n_rad),
                                        # with pf[:, :, 0] containing low frequency content and pf[:, :, -1] containing
                                        # high frequency content.
    pf[...,0] = 0
    pf /= np.linalg.norm(pf, axis=2, ord=2)[..., np.newaxis]  # Normalize each ray.
    pf_full = PolarFT.half_to_full(pf)  # pf_full has size (n_images) x (n_theta) x (n_rad)

    logger.info(f'Polar Fourier transform of {pf_full.shape[0]} projections calculated. Each '
                f'contains {pf_full.shape[1]} rays with {pf_full.shape[2]} coefficients.\n')


    ##############################################
    # Step 4: Computing the relative rotations   #
    ##############################################

    max_shift = np.ceil(img_size * max_shift_perc / 100).astype(int)
    #logger.info(f'Maximum shift is {max_shift} pixels')
    #logger.info(f'Shift step is {shift_step} pixels')
    logger.info('Computing all relative rotations...\n')

    est_rel_rots = estimate_relative_rotations(pf_full, cache_file_name, max_shift, shift_step)
    logger.info('Done computing the relative rotations\n')

    #######################################
    # Step 5: Handedness synchronization  #
    #######################################

    logger.info('Handedness synchronization...')
    u_G = jsync_TO(sym, est_rel_rots, cache_file_name)

    ##################################
    # Step 6: Rotation estimation    #
    ##################################

    logger.info('Estimating rotations')
    rots = estimate_rotations_synchronization(est_rel_rots, u_G, cache_file_name)
    rots_t = rots.transpose([0,2,1])

    ############################################
    # Step 6.5: Comparison with true rotations #
    ############################################

    logger.info("Compare with known rotations")
    mean_ang_dist = mean_angular_distance_sym(rots_t, true_rotations, sym)
    logger.info(
        f"Mean angular distance between globally aligned estimates and ground truth rotations: {mean_ang_dist}\n"
    )

    ##################################
    #  Step 7: Volume reconstruction #
    ##################################

    logger.info('Reconstructing volume')
    estimated_volume = cryo_reconstruct_TO(sym, projs, rots_t, n_r, n_theta, max_shift, shift_step, cg_max_iterations)

    ###########################
    # Step 8: Saving volume   #
    ###########################

    with mrcfile.new(outvol, overwrite=True) as mrc:
        mrc.set_data(estimated_volume)
        mrc.update_header_from_data()
    logger.info(f"Volume saved to file: {outvol}")

    return estimated_volume, rots_t