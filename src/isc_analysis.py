"""
Inter-subject correlation of heart rate (ISC-HR).

single_recording.py processes one recording into an interpolated HR series.
This file calls that once per subject, then measures how much their heart-rate
fluctuations move in sync while listening to the same stimulus.

There are also two functions which executes the hypothesis test for one combination(session,stimulus)
and across all 10 combinations at once.
"""

import numpy as np
from loading import list_subjects, ecg_paths
from single_recording import process_single_recording
from hypothesis_test import run_hypothesis_test


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
    mean_hrs : list[float]
        Each subject's mean instantaneous HR (BPM).        
    """
    if subjects is None:
        subjects = list_subjects(experiment_root)

    ids, series, mean_hrs = [], [], []
    for subj in subjects:
        tsv, js = ecg_paths(experiment_root, subj, session, stim)
        try:            
            _, hr_interp, _, n_peaks, mean_hr = process_single_recording(tsv, js)
        except (FileNotFoundError,IndexError):
            # subject numbering in BBBD is not contiguous: some files are absent
            if verbose:
                print(f"  {subj}: file missing, skipped")
            continue

        ids.append(subj)
        series.append(hr_interp)
        mean_hrs.append(mean_hr)
        if verbose:
            print(f"  {subj}: {n_peaks} peaks, mean HR {mean_hr:.1f} BPM, "
                  f"{len(hr_interp)} samples")

    return ids, series, mean_hrs


def filter_by_hr_range(ids, series, mean_hrs, low=40.0, high=160.0, verbose=False):
    """
    Automatic filter which drops subjects whose mean HR falls outside a physiologically
    plausible range (default 40-160 BPM).

    Returns
    -------
    ids_kept : list[str]
    series_kept : list[np.ndarray]
    dropped : list[tuple[str, float]]
        (subject id, mean HR) for every subject excluded.
    """
    ids_kept, series_kept, dropped = [], [], []
    for subj, s, mhr in zip(ids, series, mean_hrs):
        if low <= mhr <= high:
            ids_kept.append(subj)
            series_kept.append(s)
        else:
            dropped.append((subj, mhr))
            if verbose:
                print(f"  QC: {subj} excluded (mean HR {mhr:.1f} BPM outside "
                      f"[{low}, {high}])")
    return ids_kept, series_kept, dropped




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


def run_isc_hr_pipeline(experiment_root, session, stim, hr_low=40.0, hr_high=160.0,
                         n_perm=10000, alpha=0.05, seed=42, verbose=False):
    """
    This function run the full pipeline of processing ISC-HR across all subjects and
    does also hypothesis test for one (session,stimulus) combination.
    
    1. Collect raw data
    2. Filtering
    3. align series
    4. correlation
    5. ISC-HR
    6. hypothesis test.

    Returns
    -------
    dict with keys:
        'session', 'stim'      : str
        'ids'                  : list[str] -> subjects kept after filtering
        'dropped'              : list[tuple[str, float]] -> subjects excluded by filtering
        'isc'                  : np.ndarray -> ISC-HR per subject
        'p_values'             : np.ndarray -> hypothesis test with p-value output per subject
        'significant'          : np.ndarray of bool -> results after FDR correction
        'n_subjects'           : int -> subjects analysed (post-filtering)
        'n_significant'        : int
        'mean_isc'             : float -> group mean ISC-HR
    """
    ids_raw, series_raw, mean_hrs = collect_hr_series(
        experiment_root, session, stim, verbose=verbose)

    ids, series, dropped = filter_by_hr_range(
        ids_raw, series_raw, mean_hrs, low=hr_low, high=hr_high, verbose=verbose)

    aligned = align_series(series)
    corr = correlation_matrix(aligned)
    isc = isc_hr(corr)

    result = run_hypothesis_test(aligned, isc, n_perm=n_perm, alpha=alpha, seed=seed)

    return {
        'session': session,
        'stim': stim,
        'ids': ids,
        'dropped': dropped,
        'isc': isc,
        'p_values': result['p_values'],
        'significant': result['significant'],
        'n_subjects': len(ids),
        'n_significant': result['n_significant'],
        'mean_isc': float(isc.mean()),
    }


SESSIONS = ['ses-01', 'ses-02']   # 01 Attentive, 02 Distracted
STIMULI = ['stim01', 'stim02', 'stim03', 'stim04', 'stim05']


def run_pipeline_all_combinations(experiment_root, sessions=None, stimuli=None,
                          hr_low=40.0, hr_high=160.0, n_perm=10000, alpha=0.05,
                          seed=42, verbose=False):
    """
    This function run the full pipeline for every(session,stimulus) pair.

    Regarding the session 2 (Distracted), this functions runs the same standard
    within-group correlation as for session 1 (Attentive).
    run_isc_hr_pipeline does not reproduce the paper reference result using attentive-referenced method.

    Parameters
    ----------
    experiment_root : str
    sessions, stimuli : list[str] or None
        Defaults to the full 2x5 = 10 combinations (SESSIONS, STIMULI above).
    hr_low, hr_high : float
        filter bounds passed to filter_by_hr_range.
    n_perm, alpha, seed : 
        same as in hypothesis_test.run_hypothesis_test.
    verbose : bool
        If True, print progress for each combination as it runs.

    Returns
    -------
    list[dict]
        One result dict per combination.
    """
    sessions = sessions if sessions is not None else SESSIONS
    stimuli = stimuli if stimuli is not None else STIMULI

    results = []
    for session in sessions:
        for stim in stimuli:
            if verbose:
                print(f"=== {session}, {stim} ===")
            r = run_isc_hr_pipeline(
                experiment_root, session, stim,
                hr_low=hr_low, hr_high=hr_high,
                n_perm=n_perm, alpha=alpha, seed=seed, verbose=verbose)
            results.append(r)
            if verbose:
                print(f"  {r['n_subjects']} subjects, mean ISC-HR={r['mean_isc']:.3f}, "
                      f"{r['n_significant']}/{r['n_subjects']} significant\n")

    return results


def summary_table(results):
    """
    Reduce the list of per-combination result dicts to a compact list of
    rows, one per combination.

    Returns
    -------
    list[dict], each with: session, stim, n_subjects, n_dropped, mean_isc,
    n_significant, pct_significant
    """
    rows = []
    for r in results:
        rows.append({
            'session': r['session'],
            'stim': r['stim'],
            'n_subjects': r['n_subjects'],
            'n_dropped': len(r['dropped']),
            'mean_isc': round(r['mean_isc'], 4),
            'n_significant': r['n_significant'],
            'pct_significant': round(100 * r['n_significant'] / r['n_subjects'], 1),
        })
    return rows
