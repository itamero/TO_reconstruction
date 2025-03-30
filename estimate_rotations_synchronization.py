import pickle
import numpy as np
import logging

from jsync_TO import upper_tri_2_ind, multi_Jify

def estimate_rotations_synchronization(est_rel_rots, u_G, cache_file_name):
    """

    :param est_rel_rots:    numpy array ((n_images) x (n_images)). Entry (i,j) in the matrix corresponds
                            to Rij described above, holding the index of the estimated rotation matrix in cache R.
    :param u_G:             Handedness synchronization labels.
    :param cache_file_name: Name of the cache file that includes [R, l_self_ind, l_ij_ji_ind].
    """
    with open(cache_file_name, 'rb') as f:
        R, _, _, _, _ = pickle.load(f)

    r_candidates = len(R)
    n_images = len(est_rel_rots)

    H = np.zeros((3, 3*n_images, 3*n_images))

    # Initialize a 9x3x3 zero array
    e_kl = np.zeros((9, 3, 3))
    # Assign values
    e_kl[0, :, :] = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    e_kl[1, :, :] = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    e_kl[2, :, :] = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    e_kl[3, :, :] = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    e_kl[4, :, :] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    e_kl[5, :, :] = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    e_kl[6, :, :] = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    e_kl[7, :, :] = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    e_kl[8, :, :] = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])

    # Constructing matrix H

    # Block (0,1)
    ind = upper_tri_2_ind(0,1,n_images)
    H[0,0:3,3:6] = R[est_rel_rots[0,1]] @ e_kl[0] @ (R[est_rel_rots[1,0]]).T
    H[1,0:3,3:6] = R[est_rel_rots[0,1]] @ e_kl[4] @ (R[est_rel_rots[1,0]]).T
    H[2,0:3,3:6] = R[est_rel_rots[0,1]] @ e_kl[8] @ (R[est_rel_rots[1,0]]).T
    if u_G[ind] < 0:
        H[:, 0:3, 3:6] = multi_Jify(H[:, 0:3, 3:6] )

    # Block (0,j) for j=2,...,n_images-1
    for p_j in range(2,n_images):
        max_norm_0 = 0
        max_norm_1 = 0
        max_norm_2 = 0
        pair_ind = upper_tri_2_ind(0, p_j, n_images)
        for kl in [0, 4, 8]:
            H0j = R[est_rel_rots[0,p_j]] @ e_kl[kl] @ (R[est_rel_rots[p_j,0]]).T
            if u_G[pair_ind] < 0:
                H0j = multi_Jify(H0j)
            norm_0 = norm2(H[0, 0:3, 3:6].T @ H0j)
            if norm_0 > max_norm_0:
                max_norm_0 = norm_0
                H[0, 0:3, 3*p_j:3*p_j+3] = H0j
            norm_1 = norm2(H[1, 0:3, 3:6].T @ H0j)
            if norm_1 > max_norm_1:
                max_norm_1 = norm_1
                H[1, 0:3, 3*p_j:3*p_j+3] = H0j
            norm_2 = norm2(H[2, 0:3, 3:6].T @ H0j)
            if norm_2 > max_norm_2:
                max_norm_2 = norm_2
                H[2, 0:3, 3*p_j:3*p_j+3] = H0j

    # Block (i,j) for i=1,...,n_images-2, j=2,...,n_images-1. i<j
    for p_i in range(1,n_images-1):
        for p_j in range(p_i+1,n_images):
            pair_ind = upper_tri_2_ind(p_i, p_j, n_images)
            min_norm_0 = 10
            min_norm_1 = 10
            min_norm_2 = 10
            Hij_0 = H[0, 0:3, 3*p_i:3*p_i+3].T  @ H[0, 0:3,3*p_j:3*p_j+3]
            Hij_1 = H[1, 0:3, 3*p_i:3*p_i+3].T  @ H[1, 0:3,3*p_j:3*p_j+3]
            Hij_2 = H[2, 0:3, 3*p_i:3*p_i+3].T  @ H[2, 0:3,3*p_j:3*p_j+3]
            for kl in range(9):
                for sign in [1,-1]:
                    Hij = sign * R[est_rel_rots[p_i,p_j]] @ e_kl[kl] @ R[est_rel_rots[p_j,p_i]].T
                    if u_G[pair_ind] < 0:
                        Hij = multi_Jify(Hij)
                    if norm2(Hij_0 - Hij) < min_norm_0:
                        H[0, 3*p_i: 3*p_i+3, 3*p_j: 3*p_j+3] = Hij
                        min_norm_0 = norm2(Hij_0 - Hij)
                    if norm2(Hij_1 - Hij) < min_norm_1:
                        H[1, 3*p_i: 3*p_i+3, 3*p_j: 3*p_j+3] = Hij
                        min_norm_1 = norm2(Hij_1 - Hij)
                    if norm2(Hij_2 - Hij) < min_norm_2:
                        H[2, 3*p_i: 3*p_i+3, 3*p_j: 3*p_j+3] = Hij
                        min_norm_2 = norm2(Hij_2 - Hij)



    # Block (i,i) for i=0,...,n_images-1
    for m in [0,1,2]:
        for p_i in range(n_images):
            for p_j in range(n_images):
                if p_i < p_j:
                    H[m, 3*p_i:3*p_i+3, 3*p_i:3*p_i+3] = (H[m, 3*p_i:3*p_i+3, 3*p_i:3*p_i+3] +
                            H[m, 3*p_i:3*p_i+3, 3*p_j:3*p_j+3] @ H[m, 3*p_i:3*p_i+3, 3*p_j:3*p_j+3].T)
                if p_i > p_j:
                    H[m, 3*p_i:3*p_i+3, 3*p_i:3*p_i+3] = (H[m, 3*p_i:3*p_i+3, 3*p_i:3*p_i+3] +
                            H[m, 3*p_j:3*p_j+3, 3*p_i:3*p_i+3].T @ H[m, 3*p_j:3*p_j+3, 3*p_i:3*p_i+3])
            Hii = H[m, 3*p_i:3*p_i+3, 3*p_i:3*p_i+3] / (2*(n_images-1))
            U, S, Vt = np.linalg.svd(Hii)
            H[m, 3*p_i:3*p_i+3, 3*p_i:3*p_i+3] = S[0] * np.outer(U[:,0], Vt[0]) # First columns of U,V


    # Block (i,j) for i<j
    for m in [0,1,2]:
        H[m] = H[m] + H[m].T


    V = np.zeros((3, 3*n_images))
    for m in [0,1,2]:
        eigenvalues, eigenvectors = np.linalg.eig(H[m])
        ind = np.argsort(eigenvalues)[::-1]
        logging.info(f"5 largest eigenvalues {eigenvalues[ind[0]]:.3f}, {eigenvalues[ind[1]]:.3f}, "
                     f"{eigenvalues[ind[2]]:.3f}, {eigenvalues[ind[3]]:.3f}, {eigenvalues[ind[4]]:.3f}")
        evect1 = eigenvectors[:, ind[0]]
        for i in range(n_images):
            vi = evect1[3*i:3*i+3]
            vi = vi / norm2(vi)
            V[m, 3*i:3*i+3] = vi

    rots = np.zeros((n_images, 3, 3))
    for i in range(n_images):
        for m in [0,1,2]:
            rots[i,:,m] = V[m, 3*i:3*i+3]

    if np.linalg.det(rots[0])<0:
        rots[:,:,-1] = -rots[:,:,-1] # Multiply 3rd column of each matrix by (-1)
    return rots

def norm2(A):
    return np.linalg.norm(A, ord=2)