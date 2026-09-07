"""
significance.py — hypothesis test of ISC-HR via circular-shift permutation.


For each subject, its HR series is circularly shifted by a random amount,
which destroys the subject's temporal alignment with the group while leaving
the internal structure of its own untouched.
The ISC-HR is recomputed on this shuffled version. By repeating this 10.000 times
it builds a null distribution.
The p-value is the fraction of that null distribution at least as extreme as the real,
unshuffled ISC-HR.
The p-value counts how many of the 10.000 shuffled (null) values are as large as the real observed ISC-HR, divided by 10,000. 
For example a small p-value like 0.0001 for a subject, whose real ISC-HR of 0.136 was barely matched by chance,
means the observed synchrony is unlikely to be random.
This is done indepenently per subject, then Benjamini-Hochberg FDR correction is applied across subjects.
"""

import numpy as np


def _zscore(x):
    """
    It standardizes each row to mean 0 and standard deviation 1.

    If x has multiple rows (for example many shifted versions of a series, one per
    row), each row is standardized on its own, using its own mean and its
    own std.

    Once two series are standardized this way, their Pearson
    correlation is just their dot product divided by the number of samples.
    This lets us compute thousands of correlations at once with a single
    matrix multiplication, instead of looping and calling np.corrcoef every
    time.

    Parameters
    ----------
    x : np.ndarray
        Array to standardize. Can be one dimensional (a single series) or 
        two dimenstional (several rows, for example many shifted versions of a series, or several subjects' series stacked together). 
        If two dimensional, each row is standardized independently of the others.

    Returns
    -------
    np.ndarray, same shape as x
        The standardized array: every row has mean 0 and standard deviation 1.
    """
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / std


def _fisher_average(corrs, axis=-1):
    """
    It converts each correlation to Fisher-Z (z = arctanh).
    It average the z-values and convert the average back to a correlation (r = tanh(z)).

 
    This is the same formula used to compute ISC-HR in isc_analysis.py.
    It's needed here too because here it needs to run on a large batch at once,
    rather than on a single correlation matrix.
 
    The parameter corrs must not include self-correlation (r=1): 
    arctanh(1) is infinite, and a subject's correlation with itself carries no information anyway.

    Parameters
    ----------
    corrs : np.ndarray
        Correlation values to average. Must not include self-correlation
        (r=1): arctanh(1) is infinite, and a subject's correlation with
        itself carries no information.
    axis : int
        Which axis to average over, in Fisher-Z space, before converting
        back. Default is the last axis -1.

    Returns
    -------
    np.ndarray or float
        The averaged correlation(s), back on the original r scale (same
        shape as corrs, with `axis` removed).
    """
    z = np.arctanh(corrs)
    return np.tanh(np.mean(z, axis=axis))


def null_distribution_for_subject(subject_idx, aligned, n_perm=10000, rng=None):
    """
    This function builds the null distribution for one subject.
    It answers the following question:
    What ISC-HR values would this subject get, purely by chance, if their series were shifted in
    a circle to n_perm different random starting points in time?
    
   
    All n_perm shifted versions of the subject's series are built at once,
    using shift amounts drawn from 1 to n_samples-1 (never 0, since a shift of
    0 would leave the series unchanged and would not be a valid permutation).
    Each shift moves every value forward by that many positions, with whatever
    falls off the end reappearing at the start.
    The set of values in the series never changes, only their order does. 
    Every resulting correlation against the rest of the group is then computed in a single matrix
    multiplication rather than one at a time.


    Parameters
    ----------
    subject_idx : int
        Row index (in `aligned`) of the subject being tested.
    aligned : np.ndarray, shape (n_subjects, n_samples)
        Output of isc_analysis.align_series.
    n_perm : int
        Number of circular-shift permutations (10,000 in this case).
    rng : np.random.Generator or None
        The random number generator used to draw the shift amounts.        

    Returns
    -------
    np.ndarray, shape (n_perm,)
        Null ISC-HR values for this subject.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_subjects, n_samples = aligned.shape
    subject_series = aligned[subject_idx]
    others = np.delete(aligned, subject_idx, axis=0)   

    # one random shift amount per permutation
    shifts = rng.integers(1, n_samples, size=n_perm)

    
    idx = (np.arange(n_samples)[None, :] - shifts[:, None]) % n_samples
    shifted = subject_series[idx]  

    # Pearson correlation of every shifted version against every other subject,
    # via standardized dot product: r = (z_shifted . z_other) / n_samples
    z_shifted = _zscore(shifted)                        # (n_perm, n_samples)
    z_others = _zscore(others)                          # (n_subjects-1, n_samples)
    corrs = (z_shifted @ z_others.T) / n_samples         # (n_perm, n_subjects-1)

    return _fisher_average(corrs, axis=1)                # (n_perm,)


def permutation_test(aligned, isc_observed, n_perm=10000, seed=None, verbose=False):
    """
    This function run the circular shift premutation test for every subject and return one p-value per subject.
    For each subject, it builds a null distribution of ISC-HR values and computes a one-tailed p-value.
    
    A small p-value means it would be rare, under pure chance, to see synchrony this strong. 
    So the real result is unlikely to be random.
 
    Parameters
    ----------
    aligned : np.ndarray, shape (n_subjects, n_samples)
        Output of isc_analysis.align_series.
    isc_observed : np.ndarray, shape (n_subjects,)
        Output of isc_analysis.isc_hr (the real, unshuffled ISC-HR values).
    n_perm : int
        Number of permutations per subject (10,000 in the paper).
    seed : int or None
        Fixes the random number generator so the same result can be
        reproduced exactly on a re-run.
    verbose : bool
        If True, print each subject's result as it is computed.

    Returns
    -------
    p_values : np.ndarray, shape (n_subjects,)
        One p-value per subject, in the same order as isc_observed.
    """
    rng = np.random.default_rng(seed)
    n_subjects = aligned.shape[0]
    p_values = np.empty(n_subjects)

    for i in range(n_subjects):
        null = null_distribution_for_subject(i, aligned, n_perm=n_perm, rng=rng)
        # With the plain proportion, a subject whose observed value beats all 10,000 permutations gets
        # p = 0 exactly, which claims chance could never produce this result.
        n_exceeding = np.sum(null >= isc_observed[i])
        p_values[i] = (n_exceeding + 1) / (n_perm + 1)
        if verbose:
            print(f"  subject {i}: observed={isc_observed[i]:.3f}, "
                  f"null mean={null.mean():.3f}, p={p_values[i]:.4f}")

    return p_values


def benjamini_hochberg(p_values, alpha=0.05):
    """
    Testing many subjects at once inflates the chance of false positives.
    Even if no subject were truly synchronized, testing 27 of them at a flat alpha=0.05 each would 
    still be expected to flag about 27 x 0.05 ≈ 1.4 of them as "significant" purely by luck.
    FDR(False Discovery Rate) correction controls the expected proportion of false
    positives among the subjects called significant.

    Procedure: sort p-values ascending; find the largest rank k such that
    p(k) <= (k/n) * alpha; every p-value at or below that rank is significant.

    Parameters
    ----------
    p_values : np.ndarray, shape (n_subjects,)
    alpha : float
        The overall false discovery rate we are willing to tolerate across
        the whole batch.

    Returns
    -------
    significant : np.ndarray of bool, shape (n_subjects,)
        True where the subject's ISC-HR is significant after FDR correction,
        in the SAME order as the input p_values (not sorted).
    """
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    ranks = np.arange(1, n + 1)
    thresholds = (ranks / n) * alpha
    below = sorted_p <= thresholds

    if not np.any(below):
        return np.zeros(n, dtype=bool)

    # largest rank still satisfying the condition; everything up to it is significant
    k_max = np.max(np.where(below)[0])
    significant_sorted = np.zeros(n, dtype=bool)
    significant_sorted[:k_max + 1] = True

    # map back to original order
    significant = np.empty(n, dtype=bool)
    significant[order] = significant_sorted
    return significant


def run_hypothesis_test(aligned, isc_observed, n_perm=10000, alpha=0.05, seed=None, verbose=False):
    """
    This function run the full hypothesis test (permutation test + FDR correction).
    
    It runs the permutation test for every subject, then apply the FDR correction across all of them.
    permutation_test and benjamini_hochberg above are the two steps it executes.
    
    Returns
    -------
    dict with keys:
        'p_values'    : np.ndarray (n_subjects,) — uncorrected p-values
        'significant' : np.ndarray of bool (n_subjects,) — after FDR correction
        'n_significant': int — how many subjects pass
        'n_subjects'  : int — total subjects tested
    """
    p_values = permutation_test(aligned, isc_observed, n_perm=n_perm,
                                 seed=seed, verbose=verbose)
    significant = benjamini_hochberg(p_values, alpha=alpha)

    return {
        'p_values': p_values,
        'significant': significant,
        'n_significant': int(significant.sum()),
        'n_subjects': len(isc_observed),
    }
