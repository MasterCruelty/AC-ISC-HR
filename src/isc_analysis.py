"""
isc_analysis.py — inter-subject correlation of heart rate (ISC-HR).

single_recording.py processes one recording into an interpolated HR series.
This file calls that once per subject, then measures how much their heart-rate
fluctuations move in sync while listening to the same stimulus.
"""

import numpy as np
from loading import list_subjects, ecg_paths
from single_recording import process_single_recording


def collect_hr_series(experiment_root, session, stim, subjects=None, verbose=False):
    """
    Run the single recording pipeline on every subject for one (session, stim) and
    collect their interpolated HR series.

    Parameters
    ----------
    experiment_root : str
        Root folder of the experiment (the one containing sub-01, sub-02, ...).
    session : str
        'ses-01' (Attentive) or 'ses-02' (Distracted).
    stim : str
        Stimulus id, for example 'stim01'.
    subjects : list[str] or None
        If None, every subject found on disk is used.

    Returns
    -------
    ids : list[str]
        Subject ids actually used (only those with a valid recording).
    series : list[np.ndarray]
        One interpolated HR series per subject, in the same order as ids.
    """
    if subjects is None:
        subjects = list_subjects(experiment_root)

    ids, series = [], []
    for subj in subjects:
        tsv, js = ecg_paths(experiment_root, subj, session, stim)
        try:            
            _, hr_interp, _, n_peaks, mean_hr = process_single_recording(tsv, js)
        except FileNotFoundError:
            # subject numbering in BBBD is not contiguous: some files are absent
            if verbose:
                print(f"  {subj}: file missing, skipped")
            continue

        ids.append(subj)
        series.append(hr_interp)
        if verbose:
            print(f"  {subj}: {n_peaks} peaks, mean HR {mean_hr:.1f} BPM, "
                  f"{len(hr_interp)} samples")

    return ids, series


def align_series(series):
    """
    Truncate all HR series to the shortest common length.

    Pearson correlation needs equal-length vectors, but different recordings
    have slightly different durations, so their interpolated series differ in
    length by a few samples.

    Returns
    -------
    np.ndarray, shape (n_subjects, min_length)
    """
    min_len = min(len(s) for s in series)
    return np.array([s[:min_len] for s in series])


def correlation_matrix(aligned):
    """
    Pearson correlation coefficient between every pair of subjects.

    Returns
    -------
    np.ndarray, shape (n_subjects, n_subjects)
        Symmetric matrix; entry (i, j) is the correlation between subject i and
        subject j. The diagonal is 1 (each subject correlated with itself).
    """
    # np.corrcoef treats each row as a variable by default, which matches our
    # (n_subjects, n_samples) layout: one row per subject.
    return np.corrcoef(aligned)


def isc_hr(corr):
    """
    the ISC-HR value of each subject.

    Returns
    -------
    np.ndarray, shape (n_subjects,)
        The ISC-HR value for each subject (same order as the matrix rows).
    """
    c = corr.copy()
    np.fill_diagonal(c, np.nan)      # exclude self-correlation before transform
    z = np.arctanh(c)                # Fisher-Z of every pairwise correlation
    z_mean = np.nanmean(z, axis=1)   # mean over the rest of the group, per row
    return np.tanh(z_mean)           # inverse Fisher-Z (step 5)
