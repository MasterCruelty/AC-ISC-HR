"""
main.py — Entry point.

Chooses what to run:
  - Stage 1: process the single pilot recording and print a summary
             (a sanity check on stage 1, single_recording.py).
  - Stage 2: run the full ISC-HR analysis across all subjects
             (stage 2, isc_analysis.py).

Set EXPERIMENT_ROOT to the Experiment root on your machine.
Switch PHASE below to choose which one to run.
"""

from single_recording import process_single_recording
from isc_analysis import collect_hr_series, align_series, correlation_matrix, isc_hr

EXPERIMENT_ROOT = 'data'   # adjust to the real Experiment 2 root
PHASE = 2                  # 1 = single-recording sanity check, 2 = full ISC-HR


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


if __name__ == '__main__':
    if PHASE == 1:
        run_phase1_pilot()
    elif PHASE == 2:
        run_phase2()
