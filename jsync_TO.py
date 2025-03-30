import numpy as np
import math
import logging
import itertools
import heapq
import pickle
from numpy.linalg import norm

from cryo_TO_group_elements import cryo_TO_group_elements
from aspire.utils.random import randn


def jsync_TO(sym, est_rel_rots, cache_file_name):
    """
    Synchronize handedness using the global J-sync method.

    :param sym:                 Symmetry type: 'T' or 'O'.
    :param est_rel_rots:        Numpy array of estimated relative rotation matrices for each pair of projections.
    :param cache_file_name:     Name of pkl file containing all candidate rotations.

    :return: u_G -              Final handedness synchronization labels.
    """
    gR = cryo_TO_group_elements(sym)[0]
    n_gR = len(gR)
    n_images = len(est_rel_rots)

    # Calculate estimated relative rotation matrices
    Rijs = calc_est_rel_rotations(gR, n_images, est_rel_rots, cache_file_name)

    logging.info("Syncing relative rotations")
    # Find best J_configuration.
    J_list = J_configuration(n_gR, Rijs, n_images)

    logging.info("Done matching triplets, now running power method")
    # Determine relative handedness of Rijs.
    J_sync = J_sync_power_method(n_images, J_list)

    return J_sync


def calc_est_rel_rotations(gR, n_images, est_rel_rots, cache_file_name):
    """
    Calculate estimated relative rotation matrices for each pair of projections.


    :param n_gR:            Number of group elements.
    :param cache_file_name: Name of mat file containing all candidate rotations
    :param est_rel_rots:
    :param n_images:        Number of images.

    :return: Rijs:      An ((n-choose-2) x n_gR x 3 x 3) array where each 3x3 slice holds an estimate
                        for the corresponding outer-product Ri @ g_k @ Rj.T. Each estimate might have a
                        spurious J independently of other estimates.
    """
    n_images = len(est_rel_rots)
    with open(cache_file_name, 'rb') as f:
        R, _, _, _, _ = pickle.load(f)
    n_gR = len(gR)

    n_pairs = math.comb(n_images, 2)
    Rijs = np.zeros((n_pairs, n_gR, 3, 3))

    for i in range(n_images-1):
        for j in range(i+1, n_images):
            pair_ind = upper_tri_2_ind(i, j, n_images)
            for k in range(n_gR):
                Rijs[pair_ind,k,:,:] = R[est_rel_rots[i,j]] @ gR[k] @ (R[est_rel_rots[j,i]]).T
    return Rijs



def J_configuration(n_gR, Rijs, n_images):
    """
    List of n-choose-3 indices in {0,1,2,3} indicating
    which J-configuration for each triplet of Rijs, i<j<k.
    """
    n_trip = math.comb(n_images, 3)
    J_list = np.zeros(n_trip)

    final_votes = np.zeros(4)
    triplets_indices = lin2sub3_map(n_images)

    k1s = upper_tri_2_ind(triplets_indices[:,0], triplets_indices[:,1], n_images)
    k2s = upper_tri_2_ind(triplets_indices[:,0], triplets_indices[:,2], n_images)
    k3s = upper_tri_2_ind(triplets_indices[:,1], triplets_indices[:,2], n_images)
    ks  = np.array([k1s, k2s, k3s])
    Rijs_t = np.transpose(Rijs, [0, 1, 3, 2])

    for t in range(n_trip):
        k1 = ks[0,t]
        k2 = ks[1,t]
        k3 = ks[2,t]
        Rij = Rijs[k1]
        Rijk = np.array([Rijs[k2], Rijs_t[k3]])

        final_votes[0], prod_arr = compare_rot(n_gR, Rij, Rijk)
        final_votes[1], _        = compare_rot(n_gR, Rij, [], multi_Jify(prod_arr))
        k2_Jified = multi_Jify(Rijk[1])
        Rijk[1] = k2_Jified
        final_votes[2], prod_arr = compare_rot(n_gR, Rij, Rijk)
        final_votes[3], _ = compare_rot(n_gR, Rij, [], multi_Jify(prod_arr))
        decision = np.argmin(final_votes)
        J_list[t] = decision

    return J_list.astype(int)

def compare_rot(n_gR, Rij, Rijk, Jified_rot=None):
    if Jified_rot is None:
        prod_arr = np.matmul(Rijk[0][:, np.newaxis, :, :], Rijk[1][np.newaxis, :, :, :])
    else:
        prod_arr = Jified_rot
    arr = np.zeros((n_gR, n_gR, n_gR, 3, 3))
    for i in range(n_gR):
        arr[i] = prod_arr - np.tile(Rij[i], (n_gR,n_gR,1,1))
    arr = arr.reshape(n_gR*n_gR*n_gR, 9)
    arr = np.sum(arr ** 2, axis=1)
    smallest = heapq.nsmallest(n_gR*n_gR, arr)
    vote = sum(smallest) # We sum over the smallest n_gR*n_gR values to get a vote for this J-configuration.
    return vote, prod_arr

def J_sync_power_method(n_images, J_list):
    """
    Calculate the leading eigenvector of the J-synchronization matrix
    using the power method.

    As the J-synchronization matrix is of size (n-choose-2)x(n-choose-2), we
    use the power method to compute the eigenvalues and eigenvectors,
    while constructing the matrix on-the-fly.

    :param Rijs: (n-choose-2)x3x3 array of estimates of relative orientation matrices.

    :return: An array of length n-choose-2 consisting of 1 or -1, where the sign
        of the i'th entry indicates whether the i'th relative orientation matrix
        will be J-conjugated.
    """
    n_pairs = math.comb(n_images, 2)

    # Set power method tolerance and maximum iterations.
    epsilon = 0.01
    max_iters = 100

    # Initialize candidate eigenvectors
    vec = randn(n_pairs)
    vec = vec / norm(vec)
    residual = 1
    itr = 0

    # Power method iterations
    logging.info(
        "Initiating power method to estimate J-synchronization matrix eigenvector."
    )
    while itr < max_iters and residual > epsilon:
        itr += 1
        vec_new = signs_times_v(J_list, vec, n_images)
        vec_new = vec_new / norm(vec_new)
        residual = norm(vec_new - vec)
        vec = vec_new
        logging.info(
            f"Iteration {itr}, residual {round(residual, 5)} (target {epsilon})"
        )

    # We need only the signs of the eigenvector
    J_sync = np.sign(vec)
    J_sync = np.sign(J_sync[0]) * J_sync  # Stabilize J_sync

    return J_sync

def signs_times_v(J_list, vec, n_images):
    """
    Multiplication of the J-synchronization matrix by a candidate eigenvector.
    [From aspire abinitio.CLSymmetryD2]

    :param n_images:    number of projection images
    :param J_list:      n-choose-3 length array of indices indicating the best signs configuration.
    :param vec:         The current candidate eigenvector of length n-choose-2 from the power
        method.

    :return: New candidate eigenvector of length n-choose-2. The product of the J-sync
        matrix and vec.
    """
    new_vec = np.zeros_like(vec)
    signs_confs = np.array(
        [[1, 1, 1], [-1, 1, -1], [-1, -1, 1], [1, -1, -1]], dtype=int
    )
    trip_idx = 0
    for i in range(n_images):
        for j in range(i + 1, n_images - 1):
            ij = upper_tri_2_ind(i, j, n_images)
            for k in range(j + 1, n_images):
                ik = upper_tri_2_ind(i, k, n_images)
                jk = upper_tri_2_ind(j, k, n_images)

                best_i = J_list[trip_idx]
                trip_idx += 1

                s_ij_jk = signs_confs[best_i][0]
                s_ik_jk = signs_confs[best_i][1]
                s_ij_ik = signs_confs[best_i][2]

                # Update multiplication
                new_vec[ij] += s_ij_jk * vec[jk] + s_ij_ik * vec[ik]
                new_vec[jk] += s_ij_jk * vec[ij] + s_ik_jk * vec[ik]
                new_vec[ik] += s_ij_ik * vec[ij] + s_ik_jk * vec[jk]

    return new_vec


def upper_tri_2_ind(i, j, n):
    """
    Convert row and column indices (i, j) of an upper triangular matrix (excluding the diagonal)
    into linear indices for a vectorized representation, using 0-based indexing.

    :param i: Row indices
    :param j: Column indices
    :param n: Size of the square matrix (n x n)

    :return: Linear indices corresponding to the upper triangular element
    """
    i = np.asarray(i)
    j = np.asarray(j)

    if i.size > 0 and j.size > 0:
        ind = ((2 * n - i - 1) * i) // 2 + (j - i - 1)
    else:
        ind = np.array([])

    return ind

def lin2sub3_map(N):
    # Generate all combinations of 3 distinct numbers from 0 to N-1
    # e.g. lin2sub3_map(4) = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    idx_map = np.array([list(comb) for comb in itertools.combinations(range(N), 3)])
    return idx_map

def multi_Jify(in_array):
    """
    Applies conjugation by J = diag(1,1,-1) to an array of 3x3 rotation matrices.

    """
    Jified_rot = np.array(in_array)

    if  Jified_rot.ndim == 4:
        Jified_rot[:, :, 2, 0] *= -1
        Jified_rot[:, :, 2, 1] *= -1
        Jified_rot[:, :, 0, 2] *= -1
        Jified_rot[:, :, 1, 2] *= -1
    elif  Jified_rot.ndim == 3:
        Jified_rot[:, 2, 0] *= -1
        Jified_rot[:, 2, 1] *= -1
        Jified_rot[:, 0, 2] *= -1
        Jified_rot[:, 1, 2] *= -1
    elif  Jified_rot.ndim == 2:
        Jified_rot[2, 0] *= -1
        Jified_rot[2, 1] *= -1
        Jified_rot[0, 2] *= -1
        Jified_rot[1, 2] *= -1
    return Jified_rot
