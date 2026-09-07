"""
main.py — Entry point.

Chooses what to run:
  - Stage 1: process the single pilot recording and print a summary
             (a sanity check on stage 1, single_recording.py).
  - Stage 2: run the full ISC-HR analysis across all subjects
             (stage 2, isc_analysis.py).

  - Hypothesis test: statistical significance of the Stage-2 ISC-HR via circular-shift
               permutation (significance.py). Reuses Stage 2's output rather than
               recomputing it.

Set EXPERIMENT_ROOT to the Experiment root on your machine.
Switch PHASE below to choose which one to run.
"""

from single_recording import process_single_recording
from isc_analysis import collect_hr_series, align_series, correlation_matrix, isc_hr
from hypothesis_test import run_hypothesis_test

EXPERIMENT_ROOT = 'data'   # adjust to the real Experiment 2 root
PHASE = 3                  # 1 = single-subject pipeline
                           # 2 = full ISC-HR
                           # 3 = ISC-HR + hypothesis test


def run_phase1_pilot():
    """Stage 1 on the pilot recording only"""
    tsv = f'{EXPERIMENT_ROOT}/sub-01/ses-01/beh/sub-01_ses-01_task-stim01_recording-ecg_physio.tsv.gz'
    js = f'{EXPERIMENT_ROOT}/sub-01/ses-01/beh/sub-01_ses-01_task-stim01_recording-ecg_physio.json'

    t_common, hr_interp, condition, n_peaks, mean_hr = process_single_recording(
        tsv, js, verbose=True)
    print(f"Output grid: {len(t_common)} samples at 4 Hz, "
          f"spanning {t_common[-1] - t_common[0]:.1f} s")


def run_phase2(session='ses-01', stim='stim01'):
    """Stage 2: ISC-HR across all subjects for one (session, stim)."""
    ids, series = collect_hr_series(EXPERIMENT_ROOT, session, stim, verbose=True)
    print(f"\nCollected {len(ids)} subjects.")

    aligned = align_series(series)
    print(f"Aligned to {aligned.shape[1]} samples each.")

    corr = correlation_matrix(aligned)
    isc = isc_hr(corr)

    print("\nISC-HR per subject:")
    for subj, val in zip(ids, isc):
        print(f"  {subj}: {val:.3f}")
    print(f"\nGroup mean ISC-HR: {isc.mean():.3f}")

    return ids, aligned, corr, isc



def run_significance_test(session='ses-01', stim='stim01', n_perm=10000, alpha=0.05, seed=42):
    """
    This function execute Hypothesis test, which indicates whether each subject's ISC-HR is significant.
    It does via circular-shift permutation, with Benjamini-Hochberg FDR correction across subjects.
    """

    ids, aligned, corr, isc = run_phase2(session=session, stim=stim)
 
    print(f"\nRunning hypothesis test ({n_perm} permutations per subject)...")
    result = run_significance_test(aligned, isc, n_perm=n_perm, alpha=alpha, seed=seed)
 
    print("\nHypothesis results (FDR-corrected, alpha={:.2f}):".format(alpha))
    for subj, val, p, sig in zip(ids, isc, result['p_values'], result['significant']):
        flag = '*' if sig else ' '
        print(f"  {subj}: ISC-HR={val:.3f}, p={p:.4f} {flag}")
 
    print(f"\n{result['n_significant']} / {result['n_subjects']} subjects "
          f"significant after FDR correction.")
 
    return ids, isc, result


if __name__ == '__main__':
    if PHASE == 1:
        run_phase1_pilot()
    elif PHASE == 2:
        run_phase2()
    elif PHASE == 3:
        run_hypothesis_test()
