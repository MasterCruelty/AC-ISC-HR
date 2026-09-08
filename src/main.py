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
Set STAGES_TO_RUN below to choose which one to run.
"""

from single_recording import process_single_recording
from isc_analysis import run_isc_hr_pipeline, run_all_combinations, summary_table
from hypothesis_test import run_hypothesis_test

EXPERIMENT_ROOT = 'data'   # adjust to the real Experiment 2 root

STAGES_TO_RUN = [1, 2, 3]       # Executes all stages
#STAGES_TO_RUN = [1]             # Executes only single subject ECG processing pipeline
#STAGES_TO_RUN = [2]             # Executes only ISC-HR for all subjects and hypothesis test on a single combination(session,stimulus)

def run_stage1_pilot():
    """Stage 1: single subject ECG processing pipeline"""
    tsv = f'{EXPERIMENT_ROOT}/sub-01/ses-01/beh/sub-01_ses-01_task-stim01_recording-ecg_physio.tsv.gz'
    js = f'{EXPERIMENT_ROOT}/sub-01/ses-01/beh/sub-01_ses-01_task-stim01_recording-ecg_physio.json'

    t_common, hr_interp, condition, n_peaks, mean_hr = process_single_recording(
        tsv, js, verbose=True)
    print(f"Output grid: {len(t_common)} samples at 4 Hz, "
          f"spanning {t_common[-1] - t_common[0]:.1f} s")


def run_stage2(session='ses-01', stim='stim01'):
    """Stage 2: ISC-HR across all subjects and hypothesis test for one (session, stim) combination."""
    r = run_isc_hr_pipeline(EXPERIMENT_ROOT, session, stim,
                             n_perm=n_perm, alpha=alpha, seed=seed, verbose=True)

    print(f"\n{r['n_subjects']} subjects analysed "
          f"({len(r['dropped'])} dropped by QC).")
    print("\nISC-HR + hypothesis test per subject:")
    for subj, val, p, sig in zip(r['ids'], r['isc'], r['p_values'], r['significant']):
        flag = '*' if sig else ' '
        print(f"  {subj}: ISC-HR={val:.3f}, p={p:.4f} {flag}")
    print(f"\nGroup mean ISC-HR: {r['mean_isc']:.3f}")
    print(f"{r['n_significant']} / {r['n_subjects']} subjects "
          f"significant after FDR correction.")

    return r


def run_stage3_all_combinations(session='ses-01', stim='stim01', n_perm=10000, alpha=0.05, seed=42):
    """
    This function run hypothesis test across all 10(session, stimulus) combinations.
    """
    results = run_all_combinations(EXPERIMENT_ROOT, verbose=False)

    print("\n=== Summary across all 10 combinations ===")
    for row in summary_table(results):
        print(f"  {row['session']} {row['stim']}: "
              f"{row['n_subjects']} subjects ({row['n_dropped']} dropped by QC), "
              f"mean ISC-HR={row['mean_isc']}, "
              f"{row['n_significant']}/{row['n_subjects']} significant "
              f"({row['pct_significant']}%)")
    return results

if __name__ == '__main__':
    stages = {
        1: ('Stage 1: single subject ECG processing pipeline', run_stage1_pilot),
        2: ('Stage 2: ISC-HR and hypothesis test for one combination of (session,stimulus)', run_stage2),
        3: ('Stage 3: hypothesis test for all 10 combinations of (session,stimulus)', run_stage3_all_combinations),
    }

    for stage in STAGES_TO_RUN:
        label, func = stages[stage]
        print(f"\n{'=' * 70}")
        print(f"  {label}")
        print(f"{'=' * 70}\n")
        func()
