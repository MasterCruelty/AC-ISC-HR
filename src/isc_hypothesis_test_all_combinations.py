"""
batch_analysis.py — hypothesis test across all (session, stimulus) combinations.

It runs the group-level pipeline (collect -> QC -> align -> correlate ->
ISC-HR -> hypothesis test) once per combination of session and
stimulus, and collects the results into one structured list.
"""

from isc_analysis import (collect_hr_series, filter_by_hr_range, align_series, correlation_matrix, isc_hr)
from hypothesis_test import run_hypothesis_test

SESSIONS = ['ses-01', 'ses-02']   # 01 Attentive, 02 Distracted
STIMULI = ['stim01', 'stim02', 'stim03', 'stim04', 'stim05']


def run_one_combination(experiment_root, session, stim, hr_low=40.0, hr_high=160.0, n_perm=10000, alpha=0.05, seed=42, verbose=False):
    """
    Run the full group-level pipeline + hypothesis test for one combination (session, stimulus).

    Returns
    -------
    dict with keys:
        'session', 'stim'      : str
        'ids'                  : list[str] ; subjects kept after QC
        'dropped'               : list[tuple[str, float]] ; subjects excluded by QC
        'isc'                  : np.ndarray ; ISC-HR per subject (ids order)
        'p_values'             : np.ndarray ; R1 p-value per subject
        'significant'          : np.ndarray of bool ; after FDR correction
        'n_subjects'           : int ; subjects analysed (post-QC)
        'n_significant'        : int
        'mean_isc'             : float ; group mean ISC-HR
    """
    ids_raw, series_raw, mean_hrs = collect_hr_series(experiment_root, session, stim, verbose=verbose)

    ids, series, dropped = filter_by_hr_range(ids_raw, series_raw, mean_hrs, low=hr_low, high=hr_high, verbose=verbose)

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


def run_all_combinations(experiment_root, sessions=None, stimuli=None, hr_low=40.0, hr_high=160.0, n_perm=10000, alpha=0.05, seed=42, verbose=False):
    """
    It executes run_one_combination over every (session, stimulus) pair combination.

    Parameters
    ----------
    experiment_root : str
    sessions, stimuli : list[str] or None
        Defaults to the full 2x5 = 10 combinations (SESSIONS, STIMULI above).
    hr_low, hr_high : float
        bounds passed to filter_by_hr_range.
    n_perm, alpha, seed : as in significance.run_significance_test.
    verbose : bool
        If True, print progress for each combination as it runs.

    Returns
    -------
    list[dict]
        One result dict (see run_one_combination) per combination, in the
        order sessions x stimuli.
    """
    sessions = sessions if sessions is not None else SESSIONS
    stimuli = stimuli if stimuli is not None else STIMULI

    results = []
    for session in sessions:
        for stim in stimuli:
            if verbose:
                print(f"=== {session}, {stim} ===")
            r = run_one_combination(
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

