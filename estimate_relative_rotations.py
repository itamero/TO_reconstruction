import logging
import numpy as np
import math
from tqdm import tqdm
import pickle

from cryo_TO_create_cache import sub2ind, ind2sub

def estimate_relative_rotations(pf, cache_file_name, max_shift, shift_step):
    """
    In order to reconstruct the volume we would like to estimate the rotation matrix corresponding to each projection.
    But first we estimate the relative rotation matrices.
    For each pair of projection images (using their polar Fourier transforms), estimate the rotation matrices
    Rij and Rji satisfying the set equality {Rij*g(k)*Rji^T} = {Ri*g(k)*Rj^T} for k=1,...,n_gR
    among all rotation matrices in the candidate set of rotation matrices from the cache file.

    :param pf:              Polar Fourier transform of the images. Numpy array ((n_images) x (n_theta) x (n_rad)).
    :param cache_file_name: Name of the cache file that includes [R, l_self_ind, l_ij_ji_ind].
    :param max_shift:       Maximal 1d shift (in pixels) to search between common-lines.
    :param shift_step:      Resolution of shift estimation in pixels.

    :return:                est_rel_rots - numpy array ((n_images) x (n_images)). Entry (i,j) in the matrix corresponds
                            to Rij described above, holding the index of the estimated rotation matrix in cache R.
    """
    n_images, n_r = pf.shape[0], pf.shape[2]

    shift_phases = calc_shift_phases(n_r, max_shift, shift_step)
    n_shifts = len(shift_phases)

    with open(cache_file_name, 'rb') as f:
        R, l_self_ind, l_ij_ji_ind, _, _ = pickle.load(f)

    r_candidates = len(R)

    S_self = np.zeros((n_images, r_candidates))
    est_rel_rots = np.zeros((n_images, n_images))

    n_pairs = math.comb(n_images, 2)
    ii_inds = np.zeros(n_pairs).astype(int)
    jj_inds = np.zeros(n_pairs).astype(int)

    clmats = [0] * n_pairs
    ind = 0
    for ii in range(n_images):
        for jj in range(ii+1, n_images):
            ii_inds[ind] = ii
            jj_inds[ind] = jj
            ind = ind + 1

    # Self common lines
    logging.info("Calculating self common lines scores")
    for p_i in tqdm(range(n_images), desc="Calculating self common lines scores", miniters=0):
        pf_i = pf[p_i]
        S_self[p_i, :] = scls_correlation(n_shifts, shift_phases, pf_i, l_self_ind)

    # Common lines
    for ind in tqdm(range(n_pairs), desc="Calculating common lines scores", miniters=0):
        p_i = ii_inds[ind]
        p_j = jj_inds[ind]
        clmats[ind] = max_correlation_pair_ind(n_shifts, shift_phases, pf, p_i, p_j, S_self, r_candidates, l_ij_ji_ind)

    # Store rotation indices
    for ind in range(n_pairs):
        p_i = ii_inds[ind]
        p_j = jj_inds[ind]
        est_rel_rots[p_i, p_j] = clmats[ind][0]
        est_rel_rots[p_j, p_i] = clmats[ind][1]

    return est_rel_rots.astype(int)

def max_correlation_pair_ind(n_shifts, shift_phases, pf, p_i, p_j, S_self, r_candidates, l_ij_ji_ind):

    Corrs_cls = np.zeros(l_ij_ji_ind.shape)
    pf_i = pf[p_i]
    pf_j = pf[p_j]

    Sij = np.outer(S_self[p_i].conjugate(), S_self[p_j])
    np.fill_diagonal(Sij, 0)
    for ds in range(n_shifts):
        pf_norm_shifted = pf_j * shift_phases[ds, :]
        Corrs_pi = np.real(np.matmul(pf_i.conj(), np.transpose(pf_norm_shifted)))
        Corrs_cls_tmp = Corrs_pi[ind2sub(Corrs_pi.shape, l_ij_ji_ind)]
        Corrs_cls = np.maximum(Corrs_cls, Corrs_cls_tmp)
    cl =  np.prod(Corrs_cls, axis=1)
    c = cl.reshape(r_candidates,r_candidates)

    Sij = Sij*c
    ind1, ind2 = np.unravel_index(np.argmax(Sij), Sij.shape) # The pair with the max correlation
    return ind1, ind2



def scls_correlation(n_shifts, shift_phases, pf_i, l_self_ind):

    Corrs_scls = np.zeros(l_self_ind.shape)
    for ds in range(n_shifts):
        pf_norm_shifted = pf_i * shift_phases[ds, :]
        Corrs_pi = np.real(np.matmul(pf_i.conj(), np.transpose(pf_norm_shifted)))
        Corrs_scls_tmp = Corrs_pi[ind2sub(Corrs_pi.shape, l_self_ind)]
        Corrs_scls = np.maximum(Corrs_scls, Corrs_scls_tmp)
    return np.prod(Corrs_scls, axis=1)



def calc_shift_phases(n_r, max_shift, shift_step):
    """
    Calculates the phases of all possible shifts

    :param n_r:         Number of samples along each ray (in the radial direction)
    :param max_shift:   Maximal 1D shift (in pixels)  to search between common lines
    :param shift_step:  Resolution of shift estimation in pixels.
    """
    # Maximal shift occurs at a diagonal direction
    max_shift = math.ceil(2 * math.sqrt(2) * max_shift)
    # Number of shifts to try
    n_shifts = int(np.ceil(2 * max_shift / shift_step + 1))
    rk = np.arange(0, n_r)

    # Generate all shift phases
    shift_phases = np.zeros((n_shifts,n_r)).astype(complex)
    for shift_idx in range(n_shifts):
        shift = -max_shift + shift_step * shift_idx
        shift_phases[shift_idx] = np.exp(-2 * np.pi * 1j * rk * shift / (2*n_r-1))

    return shift_phases