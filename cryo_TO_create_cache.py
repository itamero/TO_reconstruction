import numpy as np
import logging
import math
import pickle

from cryo_TO_group_elements import cryo_TO_group_elements
from gen_rotations_grid import gen_rotations_grid

def cryo_TO_create_cache(sym, resolution, n_theta, viewing_direction, in_plane_rotation):
    """
    Creates a cache file containing:
    - All candidate rotation matrices: 'R' (n_R x 3 x 3)
    - Common lines indices: 'l_ij_ji_ind' ((n_R * n_R) x n_gR)
    - Self common lines indices: 'l_self_ind' (n_R x n_scl_pairs)

    Candidate rotations are generated on a filtered grid, based on the molecule's symmetry type.
    Common and self-common lines indices are computed between rotations and symmetry group elements.

    :param sym:                 Type of symmetry ('T' or 'O').
    :param resolution:          Number of samples per 2*pi (controls grid density).
    :param n_theta:             Angular resolution for common lines detection.
    :param in_plane_rotation:   Threshold for in-plane rotation, used in filtering.
    :param viewing_direction:   Threshold for viewing direction, used in filtering.

    :return:                    Name of the cache file that includes [R, l_self_ind, l_ij_ji_ind, resolution, n_theta].
    """
    gR, scl_inds = cryo_TO_group_elements(sym)
    logging.info("Creating candidate rotations set...")
    R = TO_candidate_rotations_set(gR, resolution, viewing_direction, in_plane_rotation)

    logging.info("Computing common lines and self common lines indices sets...")
    [l_self_ind, l_ij_ji_ind] = TO_cl_scl_inds(gR, scl_inds, n_theta, R)

    logging.info("Finished computing common lines and self common lines indices sets.")
    logging.info(f"({len(l_ij_ji_ind)} x {len(gR)}) common lines calculated "
                 f"and ({len(l_self_ind)} x {len(scl_inds)}) self common lines calculated\n")

    cache_file_name = f"cache_{sym}_symmetry_resolution_{resolution}_ntheta_{n_theta}.pkl"
    with open(cache_file_name, 'wb') as f:
        pickle.dump([R, l_self_ind, l_ij_ji_ind, resolution, n_theta], f)

    logging.info("Cache file created: " + cache_file_name + "\n")
    return cache_file_name



def TO_candidate_rotations_set(gR, resolution, viewing_direction, in_plane_rotation):
    """
    Candidate rotations are generated in an approximately equally spaced grid (based on the provided resolution)
    of rotations which is filtered based on the type of symmetry present in the molecule. The filtering is based on
    whether or not two rotations are in close proximity after one is multiplied by some symmetry group element.

    :param gR:                      Symmetry group elements: ((group order) x 3 x 3) numpy array
    :param resolution:              Number of samples per 2*pi  (see gen_rotations_grid for more details)
                                    With the standard 'viewing_direction=0.996' and 'in_plane_rotation=5' the following
                                    table shows the number of rotations generated per resolution:

                                    Resolution | Grid Rotations  | Rotations After T/O Filtering
                                    -----------|-----------------|------------------------------
                                    50         | 4,484           | 1,195 (T),    724   (O)
                                    75         | 15,236          | 1,541 (T),    854   (O)
                                    100        | 39,365          | 1,704 (T),    887   (O)
                                    150        | 129,835         | 2,084 (T),    1,068 (O)

                                    Default resolution is 75
    :param viewing_direction:       The viewing angle threshold
    :param in_plane_rotation:       The inplane rotation degree threshold

    :return:  numpy array (n_Rx3x3). Set of candidate rotation matrices after filtering.
    """
    n_gR = len(gR)
    candidates_set, _ = gen_rotations_grid(resolution) # Generate approximately equally spaced rotations with specified resolution
    n_candidates = len(candidates_set)
    logging.info(f"With resolution {resolution}, a set of {n_candidates} rotations "
                 f"was originally generated.")

    close_idx = np.zeros(n_candidates)

    for r_i in range(n_candidates - 1):
        if close_idx[r_i]:
            continue
        Ri = candidates_set[r_i]
        for r_j in range(r_i + 1, n_candidates):
            if close_idx[r_j]:
                continue
            Rj = candidates_set[r_j]
            for k in range(n_gR):
                gRj = np.dot(gR[k], Rj)
                if np.inner(Ri[:, 2],gRj[:, 2]) > viewing_direction:  # Viewing angles
                    R_inplane = np.dot(np.transpose(Ri), gRj)
                    theta = abs(np.degrees(np.arctan(R_inplane[1, 0]/R_inplane[0, 0])))
                    if theta < in_plane_rotation:     # In-plane rotation
                        close_idx[r_j] = True
    R = candidates_set[close_idx==0]

    logging.info(f"With resolution {resolution}, after filtering, a set of {len(R)} rotations "
                 f"was generated.\n")
    return R

def TO_cl_scl_inds(gR, scl_inds, n_theta, R):
    """
    Computes the set of common lines induced by all rotation matrices pairs from R
    and the set of self common lines induced by each rotation matrix from R.

    :param gR:          Symmetry group elements: 3x3x(group order) numpy array
    :param scl_inds:    Indices of the symmetry group elements for which to compute the self common lines.
    :param n_theta:     Angular resolution for common lines
    :param R:           Numpy array (n_Rx3x3) of rotation matrices
    
    :return:            l_self_ind    (n_Rx(n_scl_pairs)) numpy array of self common lines indices.
                        l_ij_ji_ind   ((n_R*n_R)xn_gR))   numpy array of common lines indices.
                        Indices correspond to the common line induced by rotation matrices Ri and Rj, 
                        in range [0,1,...,n_theta-1]. Each pair of common line indices calculated is stored
                        as a single integer between 0 and (n_theta*n_theta)-1.
    """

    n_R = len(R)
    n_gR = len(gR)
    n_scl_pairs = len(scl_inds)

    # Array of linear indices of common lines and self common lines
    l_self = np.zeros((n_R, 2, n_scl_pairs))
    l_ij_ji = np.zeros((n_R * n_R, 2, n_gR))

    for i in range(n_R):
        # Compute self common lines of Ri.
        for k in range(n_scl_pairs):
            l_self[i, 0, k], l_self[i, 1, k] = commonline_R(R[i], np.matmul(R[i],gR[scl_inds[k]]), n_theta)
        for j in range(n_R):
            if i==j:
                continue
            # Compute the common lines induced by rotation matrices Ri and Rj.
            ind = sub2ind([n_R,n_R],i,j)
            for k in range(n_gR):
                l_ij_ji[ind,0,k], l_ij_ji[ind,1,k] = commonline_R(R[i], np.matmul(R[j],gR[k]), n_theta)

    l_self_ind = sub2ind([n_theta, n_theta], l_self[:,0,:], l_self[:,1,:])
    l_ij_ji_ind = sub2ind([n_theta, n_theta], l_ij_ji[:,0,:], l_ij_ji[:,1,:])

    return l_self_ind.astype(int), l_ij_ji_ind.astype(int)


def commonline_R(Ri, Rj, L):
    """
    Compute the common line induced by rotation matrices Ri and Rj.

    :param Ri:          3x3 numpy array rotation matrix.
    :param Rj:          3x3 numpy array rotation matrix.
    :param n_theta:     Angular resolution for common lines detection.

    :return:        The indices of the common lines induced by rotations Ri and Rj.
                    Indices are integers in the range [0,1,...,n_theta-1].
    """
    Ut = np.matmul(Rj, np.transpose(Ri))

    alpha_ij = np.arctan2(Ut[2, 0], -Ut[2, 1])
    alpha_ji = np.arctan2(-Ut[0, 2], Ut[1, 2])

    pi = math.pi
    alpha_ij += pi  # Shift from [-pi,pi] to [0,2*pi]
    alpha_ji += pi

    l_ij = (alpha_ij / (2 * pi)) * L
    l_ji = (alpha_ji / (2 * pi)) * L

    l_ij = int(np.round(l_ij) % L)
    l_ji = int(np.round(l_ji) % L)

    return l_ij, l_ji


def sub2ind(matrix_shape, row, col):
    """
    Converts (row, col) indices into 1D linear index. Traversing in row order.

    :param row: integer or array of integers between 0 and matrix_shape[0]-1.
    :param col: integer or array of integers between 0 and matrix_shape[1]-1.
    :param matrix_shape: Shape of the matrix (rows, cols)
    """
    rows, cols = matrix_shape
    return cols * row + col

def ind2sub(matrix_shape, ind):
    """
    Converts 1D linear index into (row, col) indices. Traversing in row order.
    :param ind: integer or array of integers between 0 and matrix_shape[0]*matrix_shape[1]-1
    :param matrix_shape: Shape of the matrix (rows, cols)
    """
    rows, cols = matrix_shape
    row_indices = ind // rows  # Integer division to get row indices
    col_indices = ind % cols
    return row_indices, col_indices

def check_cache_file(cache_file_name, sym, rotation_resolution, n_theta):
    """
    Check provided cache file is not corrupt and matches the symmetry type.
    Note that n_theta is not checked (it is possible to change it later).

    :param cache_file_name: Name of the cache file that includes [R, l_self_ind, l_ij_ji_ind].
    :param sym:             Type of symmetry ('T' or 'O').
    """
    with open(cache_file_name, 'rb') as f:
        R, l_self_ind, l_ij_ji_ind, resolution_file, n_theta_file = pickle.load(f)
    gR, scl_inds = cryo_TO_group_elements(sym)
    n_gR = len(gR)
    n_scl_pairs = len(scl_inds)
    if resolution_file != rotation_resolution:
        raise ValueError(
            f"Provided resolution {rotation_resolution} does not match cache resolution {resolution_file}.")
    if n_theta_file != n_theta:
        raise ValueError(
            f"Provided n_theta {n_theta} does not match cache n_theta {n_theta_file}.")
    if l_self_ind.shape[1] != n_scl_pairs or l_ij_ji_ind.shape[1] != n_gR:
        raise ValueError(
            f"Provided cache_file_name '{cache_file_name}' does not match given symmetry type '{sym}'.")
    logging.info(f"Cache file contains {len(R)} candidate rotations.")
    logging.info(f"Cache file contains ({len(l_ij_ji_ind)} x {len(gR)}) common lines "
                 f"and ({len(l_self_ind)} x {len(scl_inds)}) self common lines\n")
    return




