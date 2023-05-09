import logging
import numpy as np
from .seq_tools import (get_lens, get_ohe, hamming_dist,)

############ Hamming kernel ##################


def hamming_ker_exp(seqs_x, seqs_y=None, alphabet_name='dna', bandwidth=1, lag=1):
    if seqs_y is None:
        seqs_y = seqs_x
    h_dists = hamming_dist(seqs_x, seqs_y, alphabet_name=alphabet_name, lag=lag)
    return np.exp(- h_dists/bandwidth)


def hamming_ker_dot(seqs_x, seqs_y=None, alphabet_name='dna', lag=1):
    if seqs_y is None:
        seqs_y = seqs_x
    h_dists = hamming_dist(seqs_x, seqs_y, alphabet_name=alphabet_name, lag=lag)
    x_lens = get_lens(get_ohe(seqs_x))
    y_lens = get_lens(get_ohe(seqs_y))
    max_len = np.max([np.tile(x_lens[:, None], (1, len(y_lens))),
                      np.tile(y_lens[None, :], (len(x_lens), 1))], axis=0)
    dot = max_len - h_dists
    return dot / np.sqrt(x_lens[:, None] * y_lens[None, :])


def hamming_ker_imq(seqs_x, seqs_y=None, alphabet_name='dna', scale=1, beta=1/2, lag=1):
    if seqs_y is None:
        seqs_y = seqs_x
    h_dists = hamming_dist(seqs_x, seqs_y, alphabet_name=alphabet_name, lag=lag)
    return (1+scale) ** beta / (scale + h_dists) ** beta


############ Building vector field kernels #############


def get_len_ker(kernel):
    """ Take a scalar field kernel and build a kernel such that
    k((X, Y), (X', Y')) is 0 if |X|=|Y| or |X'|=|Y'|, and equal to
    k(X, X') if |X|<|Y| and |X'|<|Y'|.
    
    Parameters:
    kernel: function
        Must be able to take a single numpy 3-D OHE set of sequences
        and return a Gram metrix, as well as two sets of sequences
        and return their comparisons.
        
    Returns:
    len_ker: function
        Takes 3-D xs, 4-D ys, and returns a Gram matrix as a vector
        field kernel.
    """
    def len_ker(xs, ys):
        seq_lens_x = get_lens(xs)
        seq_lens_y = get_lens(ys)
        num_seqs, num_muts = np.shape(ys)[:-2]
        
        dels = seq_lens_y < seq_lens_x[:, None]
        ins = seq_lens_y > seq_lens_x[:, None]
        ys_del = ys[dels, :, :]
            
        vf_ker_mat = np.zeros([num_seqs, num_muts, num_seqs, num_muts])
        xs_mat = kernel(xs)
        # Those edges that are insertions are just x-x comparisons, and +ve
        vf_ker_mat += (xs_mat[:, None, :, None]
                       * ins[None, None, :, :] * ins[:, :, None, None])
        if np.sum(dels)>0:
            # Those edges that are insertions-to-deletions are x-to-y and -ve.
            # Don't calculate x-to-y for all ys, just those that are deletions
            xy_mat_del = kernel(xs, ys_del)
            xy_mat = np.zeros([num_seqs, num_seqs, num_muts])
            xy_mat[:, dels] = xy_mat_del
            vf_ker_mat += - (xy_mat[:, None, :, :]
                             * dels[None, None, :, :] * ins[:, :, None, None])
            vf_ker_mat += - (np.transpose(xy_mat, (1, 2, 0))[:, :, :, None]
                             * ins[None, None, :, :] * dels[:, :, None, None])
            
            # Those edges that are deletions-to-deletions are y-to-y and +ve.
            yy_mat_del = kernel(ys_del, ys_del)
            yy_mat = np.zeros([num_seqs, num_muts, num_seqs, num_muts])
            cross_dels = (dels[None, None, :, :]
                          * dels[:, :, None, None]).astype(bool)
            yy_mat[cross_dels] = yy_mat_del.flatten()
            vf_ker_mat += (yy_mat
                               * dels[None, None, :, :] * dels[:, :, None, None])
        return vf_ker_mat
    return len_ker


def get_sq_ker(kernel):
    """ Take a scalar field kernel and build a kernel
    (k(X, X') + k(Y, Y'))^2 if sign(X, Y) = 1 and sign(X', Y') = 1.
    
    Parameters:
    kernel: function
        Must be able to take a single numpy 3-D OHE set of sequences
        and return a Gram metrix, as well as two sets of sequences
        and return their comparisons.
        
    Returns:
    sq_ker: function
        Takes 3-D xs, 4-D ys, and signs, and returns a Gram matrix
        as a vector field kernel. ohe shapes must be the same.
    """
    def sq_ker(xs, ys, signs):
        seq_lens_x = get_lens(xs)
        seq_lens_y = get_lens(ys)
        num_seqs, num_muts = np.shape(ys)[:-2]
        
        broadcast_x = np.tile(xs[:, None], (1, num_muts, 1, 1))
        # l_seqs are X if s = 1 and Y otherwise, u_seqs are opposite
        l_seqs = np.copy(ys)
        u_seqs = np.copy(ys)
        l_seqs[signs == 1] = broadcast_x[signs == 1]
        u_seqs[signs == -1] = broadcast_x[signs == -1]
        vf_ker_mat = (kernel(l_seqs) + kernel(u_seqs))**2
        vf_ker_mat *= signs[None, None, :, :] * signs[:, :, None, None]
        return vf_ker_mat
    return sq_ker


def vs_kern(coerce_ker, delta_ker, sign_ker=None):
    """ Takes two kernels to create a coercive and deltable
    vector space kernel. Does so by summing a len_kernel and a sq_kernel.
    
    Parameters:
    coerce_ker: function
        Scalar field kernel to build len_kernel.
    delta_ker: function
        Scalar field kernel to build sq_kernel.
    sign_ker: function, default = None
        Optional kernelto use k^\nabla to determine signs between
        sequences of the same len.
    
    Returns:
    vf_ker: function
        Takes 3-D xs, 4-D ys, and signs, and returns a Gram matrix
        as a vector field kernel. ohe shapes must be the same.
    """
    ker_1 = get_len_ker(coerce_ker)
    ker_2 = get_sq_ker(delta_ker)
    def vf_ker(xs, ys, signs):    
        if sign_ker is not None:
            base_seq_x = xs[0] * 0
            base_seq_y = np.copy(xs[0])
            base_seq_y[1:] = 0
            sign_ker_mat = (sign_ker(base_seq_x[None], xs)[0, :, None]
                            + sign_ker(base_seq_y[None], ys)[0]
                            - sign_ker(base_seq_x[None], ys)[0]
                            - sign_ker(base_seq_y[None], xs)[0, :, None])
            assert np.all(sign_ker_mat != 0)
            signs = sign_ker_mat > 0
        return ker_1(xs, ys) + ker_2(xs, ys, signs)
    return vf_ker
