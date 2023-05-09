import logging
import numpy as np
import time
from scipy.special import logsumexp


def get_mutations(input_seqs, only_subs=False):
    """For each seq, return [len_seq, alphabet_size] subs,
    [len_seq+1, alphabet_size] insertions, [len_seq] deletions and concatenate them.
    Turn invalid muts into nans.
    
    Parameters:
    input_seqs: numpy array
        Sequences in OHE representation. Can have any number of dims. Must not have
        a stop. Assumes last entry of OHE representation is empty (raises error if not).
    only_subs: bool or int, default = False
        Only return substitutions. If int, only return substitutions within only_subs
        in the cyclic nbh.

    Returns:
    all_seqs: numpy array
        Same OHE length as original, so if not only_subs, make sure input seqs has extra
        OHE column (raises warning if not).
    signs: numpy array
        Signs from lexicographic ordering: 1 for deletions, -1 for insertions,
        1 if substitution is > wild-type and -1 otherwise.
    """
    assert np.sum(input_seqs[..., -1, :]) == 0, "No extra entry in OHE!"
    shape = np.shape(input_seqs)[:-2]
    seqs = input_seqs.reshape((-1,) + np.shape(input_seqs)[-2:]).astype(float)
    num_seqs, seq_len, alphabet_size = np.shape(input_seqs)
    assert alphabet_size > 1 or not only_subs, "no subs for single letter alphabet!"
    seq_len = seq_len - 1
    empty = np.sum(seqs, axis=-1) == 0 # empty positions
    empty_for_ins = np.concatenate(
        [np.zeros([num_seqs, 1], dtype=bool), empty[:, :-1]], axis=-1)

    # First substitutions.
    if alphabet_size > 1:
        # take only nbh if only_subs is an int
        if not isinstance(only_subs, bool):
            nbh = np.r_[np.arange(only_subs),
                        alphabet_size - 2 - np.arange(only_subs)]
            nbh = np.isin(range(alphabet_size-1), nbh)
        else:
            nbh = np.s_[:]
        perm_mat = np.eye(alphabet_size)
        perm_mat = np.r_[perm_mat[1:], perm_mat[[0]]]
        substitutions = np.array([[
            np.concatenate([seqs[:, :i, :],
                            np.einsum('nb,bk->nk', seqs[:, i],
                                      np.linalg.matrix_power(perm_mat, b+1))[:, None, :],
                            seqs[:, i+1:, :]], axis=-2)
            for b in range(alphabet_size-1)] for i in range(seq_len)])
        substitutions = np.transpose(substitutions, [2, 0, 1, 3, 4])[:, :, nbh, :, :]
        substitutions[empty[:, :-1]] = np.nan # substitutions in empty positions are nan'ed
        substitutions=substitutions.reshape([num_seqs, -1, seq_len+1, alphabet_size])
        sub_signs = np.cumsum(seqs[..., ::-1], axis=-1)[..., ::-1]
        sub_signs = sub_signs[..., 1:][..., ::-1]
        sub_signs[np.tile(empty[..., None], len(np.shape(empty))*(1,)
                          +(np.shape(sub_signs)[-1],))] = np.nan
        sub_signs = sub_signs[..., :-1, :][..., nbh]
        sub_signs = 2 * sub_signs.reshape([num_seqs, -1]) - 1
    else:
        substitutions = []
        sub_signs = []

    if not only_subs:
        # Then deletions.
        deletions = np.array([
            np.concatenate([seqs[:, :i, :], seqs[:, i+1:, :],
                            np.tile(np.zeros(alphabet_size)[None, None, :],
                                    (num_seqs, 1, 1))], axis=-2)
            for i in range(seq_len)])    
        deletions = np.transpose(deletions, [1, 0, 2, 3])
        deletions[empty[:, :-1]] = np.nan
        del_signs = np.ones(np.shape(deletions)[:-2])
        del_signs[empty[:, :-1]] = np.nan

        # Finally, insertions
        insertions = np.array([[
            np.concatenate([seqs[:, :i, :],
                            np.tile(base[None, None, :], (num_seqs, 1, 1)),
                            seqs[:, i:-1, :]], axis=-2)
            for base in np.eye(alphabet_size)] for i in range(seq_len+1)])
        insertions = np.transpose(insertions, [2, 0, 1, 3, 4])
        insertions[empty_for_ins] = np.nan
        ins_signs = - np.ones(np.shape(insertions)[:-2])
        ins_signs[empty_for_ins] = np.nan
        insertions = insertions.reshape([num_seqs, -1, seq_len+1, alphabet_size])
        ins_signs = ins_signs.reshape([num_seqs, -1])
    
    # Concatenate.
    if only_subs:
        all_seqs = substitutions
        all_signs = sub_signs
    else:
        all_seqs = np.concatenate([substitutions]*(alphabet_size>1)
                                  + [insertions, deletions], axis=1).reshape(
            shape + (-1,) + np.shape(input_seqs)[-2:])
        all_signs = np.concatenate([sub_signs]*(alphabet_size>1)
                                   + [ins_signs, del_signs], axis=1)
    return all_seqs, all_signs

    
def sample_mut(mut_logit, ys, signs, num_mut_samples, unif_sample=False):
    """ Sample mutations given likelihoods using Gumbel softmax trick.
    
    Parameters:
    mut_logit: numpy array
        Last axis is logits for sampling sequences ys. nans are -inf.
    ys: numpy array
        OHE of neighbours. must be same shape as mut_logit + 
        OHE seq shape.
    signs: numpy array
        Signs of ys.
    num_mut_samples: int
        Number of samples to take.
    unif_sample: bool, default = False
        Ignore logits and sample uniformly.
    
    Returns:
    select_mut: numpy array
        Sampled mutant seqs as OHEs.
    select_signs: numpy array
        Signs of sampled sequences.
    """
    select_mut = np.random.gumbel(size=np.r_[np.shape(mut_logit), [num_mut_samples]])
    select_mut += np.nan_to_num((not unif_sample)*mut_logit[..., None], nan=-np.inf)
    select_mut = np.argmax(select_mut, axis=1) #[num_seqs, num_mut_samples]
    select_mut_logit = np.array([logits[select]
                                 for logits, select in zip(mut_logit, select_mut)])
    select_signs = np.array([ss[select] for ss, select in zip(np.array(signs), select_mut)])
    select_mut = np.array([muts[select] for muts, select in zip(ys, select_mut)])
    return select_mut, select_signs


def get_sampled_mut(xs, energy_func, num_mut_samples, chi_alpha=1,
                    unif_sample=False, only_subs=False,
                    return_signs=False, return_all_muts=False):
    """ Samples mutations from underlying stochastic process from the Zanella operator.
    Start at OHE sequences xs, sample num_mut_samples from Zanella operator with
    chi(t) = t^alpha if t<1 and t^(1-alpha) if t>1.
    
    Parameters:
    xs: numpy array
        OHE sequences, can have any number of dims but must have no stop and an empty column.
    energy_func: function
        Must take OHE sequences of any number of dimensions and return log probs.
        Must return nans or -np.inf for nan seqs.
    num_mut_samples: int
    chi_alpha: float, default = 1
    unif_sample: bool, default = False
        Sample uniformly from neighbours.
    only_subs: bool, default = False
        Sample only from substitutions.
    return_signs: bool, default = False
        Return lexicographic ordering of sampled mutants.
    return_all_muts: bool, default = False
        Also return all neighbours of each sequence. Includes nan'ed sequences if
        some xs are of different length or if there is more than one empty column.

    Returns:
    select_mut: numpy array
        Shape is xs.shape + (num_mut_samples,) + xs_seq_shape.
    c_vec: numpy_array
        Partition or flux function at each x.
    ys: numpy array, optional
        All mutations for all xs. may included nan'ed sequences.
        Only provided if return_all_muts.
    mut_logits: numpy array, optional
        Logits of underlying stochastic process for all ys. Only provided if
        return_all_muts.
    select_signs: numpy array, optional
        Signs of sampled mutations. Only provided if return_signs.
    signs: numpy array, optional
        Signs of all neighbours. Only provided if return_all_muts and return_signs.
    """
    ys, signs = get_mutations(xs, only_subs)
    # Get probs.
    t0 = time.perf_counter()
    probs_x = energy_func(xs)
    logging.info("done xs probs calc: {}".format(time.perf_counter()-t0)) 
    logging.info("num_muts: {}".format(np.prod(np.shape(ys)[:-2])))
    t0 = time.perf_counter()
    probs_y = np.nan_to_num(energy_func(ys), nan=-np.inf)
    logging.info("done mut probs calc: {}".format(time.perf_counter()-t0))
    # Select muts.
    mut_logit = (probs_y - probs_x[:, None])
    mut_logit[mut_logit < 0] *= chi_alpha
    mut_logit[mut_logit > 0] *= 1 - chi_alpha
    c_vec = logsumexp(mut_logit, axis=-1)
    select_mut, select_signs = sample_mut(mut_logit, ys, signs,
                                          num_mut_samples, unif_sample)
    return ((select_mut, c_vec,) + return_all_muts * (ys, mut_logit)
            + return_signs * (select_signs,) + return_signs * return_all_muts * (signs,))
