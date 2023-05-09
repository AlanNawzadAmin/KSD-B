import logging
import time
import numpy as np
from .mutate_seqs import get_sampled_mut
from .seq_tools import pad_seq_len, set_ohe_pad
epsilon = 1e-3


def get_diff_mat(kernel, kernel_tilting_func, seqs, ys):
    """ Get Gram matrix of k^\nabla for a scalar field kernel k.
    
    Parameters:
    kernel: function
        Must be able to take in 3-dimensional OHE sequences and
        output a Gram matrix.
    kernel_tilting_function: function
        To tilt the kernel. Must be able to take 3-dimensional OHE sequences.
    seqs: numpy array
        3 dimensional OHE. Length is "num_seqs".
    ys: numpy array
        4 dimensional OHE which is [num_seqs, "num_muts"] + OHE shape.
    
    Returns:
    k_diff: numpy array
        Gram matrix, shaped [num_seqs, num_muts, num_seqs, num_muts].
    """
    # Format sequences
    num_seqs = len(seqs)
    ohe_shape = np.shape(seqs)[-2:]
    num_muts = np.shape(ys)[-3]
    flat_ys = ys.reshape((-1,) + ohe_shape)
    all_seqs_flat = np.concatenate([seqs, flat_ys])
    
    # Get Gram matrices
    t0 = time.perf_counter()
    tilt = kernel_tilting_func(all_seqs_flat)
    all_ker_mat = kernel(all_seqs_flat) * np.outer(tilt, tilt)
    logging.info("done kernel calc: {}".format(time.perf_counter()-t0)) 
    k_xx = all_ker_mat[:num_seqs, :num_seqs]
    k_xy = all_ker_mat[:num_seqs, num_seqs:].reshape(
        [num_seqs, num_seqs, num_muts])
    k_yx = np.transpose(k_xy, [1, 2, 0])
    k_yy = all_ker_mat[num_seqs:, num_seqs:].reshape(
        [num_seqs, num_muts, num_seqs, num_muts])
    k_diff = np.nan_to_num(k_yy - k_xy[:, None, :, :]
                           - k_yx[:, :, :, None] + k_xx[:, None, :, None])
    return k_diff
    
    
def get_ksd_mat(seqs, energy_func, kernel, num_mut_samples,
                kernel_tilting_func=None, vf_kernel=False, exact_ksd=False,
                only_subs=False, chi_alpha=1):
    """ Get the matrix of sequence-to-sequence terms in the KSD.
    
    Parameters:
    seqs: numpy array
        3 dimensional OHE.
    energy_func: function
        Must be able to take any-dimensional numpy OHE sequences and output log
        probabilities. Must return nan or -infty for sequences with all nans.
    kernel: function
        Must be able to take in 3-dimensional OHE sequences and
        output a Gram matrix. Must return some value for nan seqs.
    num_mut_samples: int
        Number of samples to take to approximate the sum in the KSD.
    kernel_tilting_function: function, default = None
        To tilt the kernel in the case that it is a scalar field kernel.
        Must be able to take 3-dimensional OHE sequences.
    vf_kernel: bool, default = False
        Use a vector field kernel (kernel_tilting_function) is ignored.
    exact_ksd: bool, default = False
        Do not approximate the sum in the KSD (takes much much longer).
    only_subs: bool, default = False
        Only consider substitutions, not deletions or insertions,
        in the mutation graph.
    chi_alpha: float, default = 1
        
    Returns:
    ksd_mat: numpy array
        Matrix for which ksd_mat.average() is the KSD.
    """
    assert len(np.shape(seqs)) == 3, "Seqs must be flat OHE."
    seqs = set_ohe_pad(seqs, 1)
    
    # Format mutations of seqs (ys)
    ys, c_vec, all_ys, mut_logits, signs, all_signs = get_sampled_mut(
        seqs, energy_func, num_mut_samples, unif_sample=False,
        only_subs=only_subs, return_all_muts=True, return_signs=True,
        chi_alpha=chi_alpha)
    use_ys = ys if not exact_ksd else all_ys
    use_signs = signs if not exact_ksd else all_signs
    # Log perplexities of transitions
    lls = mut_logits - c_vec[:, None]
    perps = np.exp(-np.einsum('ij,ij->i', np.nan_to_num(lls), np.exp(lls)))
    logging.info("transition perplexities: {}".format(perps))
    
    # Get kernel Gram matrix
    if vf_kernel:
        ker_mat = kernel(seqs, use_ys, use_signs)
    else:
        ker_mat = get_diff_mat(kernel, kernel_tilting_func, seqs, use_ys)
    ker_mat = np.nan_to_num(ker_mat)

    # Get probs of underlying stoch process and multiply to get ksd mat
    if exact_ksd:
        probs = np.nan_to_num(np.exp(mut_logits - c_vec[:, None]), 0)
    else:
        probs = np.ones(np.shape(ker_mat)[:2])
        probs /= np.sum(probs)
    ksd_mat = np.einsum('i,j,ik,jl,ikjl->ij',
                        np.exp(c_vec), np.exp(c_vec), probs, probs, ker_mat)
    return ksd_mat
