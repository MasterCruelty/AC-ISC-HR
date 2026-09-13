"""
Stage 1: process the single pilot recording and print a summary

Stage 2: run the full ISC-HR analysis across all subjects

Stage 3(Hypothesis test): statistical significance of the ISC-HR via circular-shift permutation.

Stage 4(Attentive-referenced correlation): does attention modulate ISC-HR? 
         Attentive-referenced correlation for the distracted condition, then a paired comparison across the 5 stimuli.


Set EXPERIMENT_ROOT to the Experiment root on your machine.
Set STAGES_TO_RUN below to choose which one to run.
"""

from single_recording import process_single_recording
from isc_analysis import run_isc_hr_pipeline, run_pipeline_all_combinations, summary_table
from hypothesis_test import run_hypothesis_test
from attention_comparison import run_attention_modulation

EXPERIMENT_ROOT = 'data'   # adjust to the real Experiment 2 root

STAGES_TO_RUN = [1, 2, 3, 4]       # Executes all stages
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


def run_stage2(session='ses-01', stim='stim01', n_perm=10000, alpha=0.05, seed=42):
    """
    Stage 2: ISC-HR across all subjects and hypothesis test for one (session, stim) combination.
    """

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


def run_stage3_all_combinations():
    """
    This function run hypothesis test across all 10(session, stimulus) combinations.
    """
    results = run_pipeline_all_combinations(EXPERIMENT_ROOT, verbose=False)

    print("\n=== Summary across all 10 combinations ===")
    for row in summary_table(results):
        print(f"  {row['session']} {row['stim']}: "
              f"{row['n_subjects']} subjects ({row['n_dropped']} dropped by QC), "
              f"mean ISC-HR={row['mean_isc']}, "
              f"{row['n_significant']}/{row['n_subjects']} significant "
              f"({row['pct_significant']}%)")
    return results


def run_stage4_attention_modulation(n_perm=10000, alpha=0.05, seed=42):
    """
    This function run the attentive-referenced correlation for the distracted condition.
    Then it does a paired comparison across all 5 stimulus.
    """
    result = run_attention_modulation(EXPERIMENT_ROOT, n_perm=n_perm, alpha=alpha, seed=seed, verbose=False)

    print("\n=== Per-subject significance (attentive-referenced method) ===")
    for r in result['per_stim']:
        att_r = r['attentive_result']
        dis_r = r['distracted_result']
        print(f"  {r['stim']}:")
        print(f"    attentive:  mean ISC-HR={r['attentive_isc'].mean():.4f}, "
            f"{att_r['n_significant']}/{att_r['n_subjects']} significant "
            f"({100*att_r['n_significant']/att_r['n_subjects']:.1f}%)")
        print(f"    distracted: mean ISC-HR={r['distracted_isc'].mean():.4f}, "
            f"{dis_r['n_significant']}/{dis_r['n_subjects']} significant "
            f"({100*dis_r['n_significant']/dis_r['n_subjects']:.1f}%)")

    print("\n=== Attention modulation of ISC-HR ===")
    print(f"Shapiro-Wilk normality check: "
          f"statistic={result['shapiro_stat']:.4f}, p={result['shapiro_p']:.4f}")
    print(f"Test used for all 5 stimuli: {result['test_used']}\n")

    for t in result['per_stim_tests']:
        print(f"  {t['stim']}: n={t['n']}, statistic={t['statistic']:.4f}, p={t['p_value']:.4f}")

    return result

#############################################
#           MAIN    
#############################################
if __name__ == '__main__':
    stages = {
        1: ('Stage 1: single subject ECG processing pipeline', run_stage1_pilot),
        2: ('Stage 2: ISC-HR and hypothesis test for one combination of (session,stimulus)', run_stage2),
        3: ('Stage 3: hypothesis test for all 10 combinations of (session,stimulus)', run_stage3_all_combinations),
        4: ('Stage 4: attention modulation of ISC-HR', run_stage4_attention_modulation),
    }

    for stage in STAGES_TO_RUN:
        label, func = stages[stage]
        print(f"\n{'=' * 70}")
        print(f"  {label}")
        print(f"{'=' * 70}\n")
        func()
