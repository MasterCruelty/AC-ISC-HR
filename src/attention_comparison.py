"""
attention_comparison.py --> reproducing R2 by veryfing if attention modulate the ISC-HR.

Considering the paper we're using as reference, ISC-HR for both the
attentive and the distracted condition is computed against the attentive
group's HR as a fixed reference, not within each condition separately.

For the distracted side, each subject is correlated against the fixed
attentive group instead of other distracted subjects. 

The main scope here is the statistical comparison of attentive vs. distracted 
ISC-HR across subjects, for each of the 5 stimuli:
First of all, a Shapiro-Wilk normality check on the paired differences is made,
followed by a paired t-test or a Wilcoxon signed-rank test depending on
that result.

In addition, an ANOVA test is computed to make the obtained results more solid.
"""

import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon
from statsmodels.stats.anova import AnovaRM
import pandas as pd

from isc_analysis import (collect_hr_series, filter_by_hr_range, align_series,
                          correlation_matrix, isc_hr, STIMULI,
)
from hypothesis_test import zscore, fisher_average, run_hypothesis_test


def isc_hr_referenced(target_ids, target_aligned, reference_ids, reference_aligned):
    """
    ISC-HR of each target subject, computed against a fixed reference group (the attentive group).

    For every target subject, this correlates their series against every
    series in the reference group, then averages those correlations in Fisher-Z space.

    Parameters
    ----------
    target_ids : list[str]
    target_aligned : np.ndarray
        shape (n_target, n_samples)
    reference_ids : list[str]
    reference_aligned : np.ndarray
        shape (n_reference, n_samples)

    Returns
    -------
    np.ndarray
        shape (n_target,)
    """
    n_samples = target_aligned.shape[1]
    z_target = zscore(target_aligned)
    z_reference = zscore(reference_aligned)
    corrs = (z_target @ z_reference.T) / n_samples   # (n_target, n_reference)

    result = np.empty(len(target_ids))
    for i, tid in enumerate(target_ids):
        row = corrs[i]
        if tid in reference_ids:
            j = reference_ids.index(tid)
            row = np.delete(row, j)   # exclude own attentive recording
        result[i] = fisher_average(row, axis=0)
    return result


def reference_groups_excluding_self(target_ids, reference_ids, reference_aligned):
    """
    This function build, for each target subject, the reference array to test them
    against under the hypothesis test. This is the same fixed reference group used in isc_hr_referenced.
    This is 'other_list' that permutation_test function expects in hypothesis_test.py.

    Returns
    -------
    list[np.ndarray]
    """
    others_list = []
    for tid in target_ids:
        if tid in reference_ids:
            j = reference_ids.index(tid)
            others_list.append(np.delete(reference_aligned, j, axis=0))
        else:
            others_list.append(reference_aligned)
    return others_list


def run_attention_comparison_one_stim(experiment_root, stim, hr_low=40.0, hr_high=160.0,
                                       n_perm=10000, alpha=0.05, seed=42, verbose=False):
    """
    This function runs the pipeline by collecting both conditions, 
    applying quality control filtering, aligning them together, 
    then computing ISC-HR differently per condition.
    Standard within-group for attentive, attentive-referenced for distracted. 
    It then runs the hypothesis test on both.

    Returns
    -------
    dict with keys:
        'stim'                                  : str
        'attentive_ids', 'distracted_ids'       : list[str] (post-QC)
        'attentive_isc', 'distracted_isc'       : np.ndarray
        'attentive_result', 'distracted_result' : dict (see run_hypothesis_test)
    """

    #attentive condition
    att_ids_raw, att_series_raw, att_hr = collect_hr_series(
        experiment_root, 'ses-01', stim, verbose=verbose)
    att_ids, att_series, _ = filter_by_hr_range(
        att_ids_raw, att_series_raw, att_hr, low=hr_low, high=hr_high, verbose=verbose)

    #distracted condition
    dis_ids_raw, dis_series_raw, dis_hr = collect_hr_series(
        experiment_root, 'ses-02', stim, verbose=verbose)
    dis_ids, dis_series, _ = filter_by_hr_range(
        dis_ids_raw, dis_series_raw, dis_hr, low=hr_low, high=hr_high, verbose=verbose)

    #There is need to align both groups together so they share the same sample length.
    #This is required before correlating across the two conditions.
    n_att = len(att_series)
    combined_aligned = align_series(att_series + dis_series)
    aligned_att = combined_aligned[:n_att]
    aligned_dis = combined_aligned[n_att:]

    # Attentive: standard within-group correlation 
    corr_att = correlation_matrix(aligned_att)
    isc_att = isc_hr(corr_att)
    result_att = run_hypothesis_test(aligned_att, isc_att, n_perm=n_perm, alpha=alpha, seed=seed)

    # Distracted: attentive-referenced correlation.
    isc_dis = isc_hr_referenced(dis_ids, aligned_dis, att_ids, aligned_att)
    others_list_dis = reference_groups_excluding_self(dis_ids, att_ids, aligned_att)
    result_dis = run_hypothesis_test(aligned_dis, isc_dis, n_perm=n_perm, alpha=alpha,
                                      seed=seed, others_list=others_list_dis)

    return {
        'stim': stim,
        'attentive_ids': att_ids,
        'distracted_ids': dis_ids,
        'attentive_isc': isc_att,
        'distracted_isc': isc_dis,
        'attentive_result': result_att,
        'distracted_result': result_dis,
    }


def paired_differences(one_stim_result):
    """
    For each subject who has a kept recording in both conditions for one_stim_result,
    compute attentive_isc - distracted_isc.

    Example: if sub-01 and sub-03 are the only subjects present in both
    attentive_ids and distracted_ids, they only appear in the output.
    sub-02 (attentive only) and sub-05 (distracted only) are dropped.

    Parameters
    ----------
    one_stim_result : dict
        Result of the (attentive, distracted) comparison for one stimulus.
        (the dict returned by run_attention_comparison_one_stim).

    Returns
    -------
    ids : list[str]
        subjects present in both conditions
    diffs : np.ndarray
        attentive_isc - distracted_isc
    """
    att_ids = one_stim_result['attentive_ids']
    dis_ids = one_stim_result['distracted_ids']
    att_isc = one_stim_result['attentive_isc']
    dis_isc = one_stim_result['distracted_isc']

    common_ids = [i for i in att_ids if i in dis_ids]
    diffs = np.array([
        att_isc[att_ids.index(i)] - dis_isc[dis_ids.index(i)]
        for i in common_ids
    ])
    return common_ids, diffs


def run_attention_modulation(experiment_root, stimuli=None, hr_low=40.0, hr_high=160.0,
                             n_perm=10000, alpha=0.05, seed=42, verbose=False):
    """
    This function run comparison_one_stim for every stimulus,
    then decide once whether to use a paired t-test or a Wilcoxon signed-rank test,
    applying the same choice to every stimulus individually.

    Returns
    -------
    dict with keys:
        'per_stim'        : list[dict] -> one result per stimulus
        'shapiro_stat', 'shapiro_p' : float -> pooled normality check
        'test_used'       : str -> 'paired t-test' or 'Wilcoxon signed-rank'
        'per_stim_tests'  : list[dict] -> {'stim', 'n', 'statistic', 'p_value'} per stimulus
    """
    stimuli = stimuli if stimuli is not None else STIMULI

    per_stim = []
    all_diffs = []
    for stim in stimuli:
        if verbose:
            print(f"=== Attention comparison: {stim} ===")
        r = run_attention_comparison_one_stim(
            experiment_root, stim, hr_low=hr_low, hr_high=hr_high,
            n_perm=n_perm, alpha=alpha, seed=seed, verbose=verbose)
        per_stim.append(r)
        _, diffs = paired_differences(r)
        all_diffs.append(diffs)
        if verbose:
            print(f"  attentive mean ISC-HR={r['attentive_isc'].mean():.3f}, "
                  f"distracted (referenced) mean ISC-HR={r['distracted_isc'].mean():.3f}\n")

    #Shapiro-Wilk test
    pooled_diffs = np.concatenate(all_diffs)
    shapiro_stat, shapiro_p = shapiro(pooled_diffs)
    use_ttest = shapiro_p > alpha   # normality not rejected -> parametric test is fine

    #based on the result of shapiro-wilk test, choice of t-test or wilcoxon is made.
    test_used = 'paired t-test' if use_ttest else 'Wilcoxon signed-rank'

    #applying the chosen test individually for every stimulus
    per_stim_tests = []
    for stim, r in zip(stimuli, per_stim):
        ids, diffs = paired_differences(r)
        if use_ttest:
            stat, p = ttest_rel(diffs, np.zeros_like(diffs))
        else:
            stat, p = wilcoxon(diffs, method='exact')
        per_stim_tests.append({'stim': stim, 'n': len(ids), 'statistic': float(stat), 'p_value': float(p)})

    return {
        'per_stim': per_stim,
        'shapiro_stat': float(shapiro_stat),
        'shapiro_p': float(shapiro_p),
        'test_used': test_used,
        'per_stim_tests': per_stim_tests,
    }




def build_long_format(per_stim):
    """
    This function reshapes the run_attention_modulation's results 
    in order to use AnovaRM which need that every subject must have a value for
    every stimulus x condition combination.

    Parameters
    ----------
    per_stim : list[dict]
        Per-stimulus results, which is result['per_stim'] from
        run_attention_modulation. Each dict must contain 'stim' (stimulus
        name), 'attentive_ids' and 'distracted_ids' (lists of subject ids),
        'attentive_isc' and 'distracted_isc' (the ISC-HR of each subject).


    Returns
    -------
    df : pandas.DataFrame
         columns: subject, stimulus, condition, isc_hr
         only subjects present in all combinations are kept.
    dropped : list[str]
        subjects excluded because their design was incomplete.
    """    

    rows = []
    for r in per_stim:
        for sid, val in zip(r['attentive_ids'], r['attentive_isc']):
            rows.append({'subject': sid, 'stimulus': r['stim'], 'condition': 'attentive', 'isc_hr': val})
        for sid, val in zip(r['distracted_ids'], r['distracted_isc']):
            rows.append({'subject': sid, 'stimulus': r['stim'], 'condition': 'distracted', 'isc_hr': val})

    df = pd.DataFrame(rows)

    n_combinations = df[['stimulus', 'condition']].drop_duplicates().shape[0]
    counts = df.groupby('subject').size()
    complete_subjects = counts[counts == n_combinations].index
    dropped = sorted(set(df['subject']) - set(complete_subjects))

    df_balanced = df[df['subject'].isin(complete_subjects)].reset_index(drop=True)
    return df_balanced, dropped


def run_anova_r2(per_stim, verbose=False):
    """
    This function run ANOVA (attention x stimulus),
    reporting fixed effects for attention and video.

    Parameters
    ----------
    per_stim : list[dict]
        Per-stimulus results, which is result['per_stim'] from run_attention_modulation.


    Returns
    -------
    dict with keys:
        'table'      : the AnovaRM result table
        'dropped'    : list[str] --> subjects dropped for an incomplete design
        'n_subjects' : int --> subjects retained
    """

    df, dropped = build_long_format(per_stim)
    if verbose and dropped:
        print(f"  ANOVA: {len(dropped)} subject(s) dropped for incomplete design: {dropped}")

    result = AnovaRM(df, depvar='isc_hr', subject='subject',
                      within=['condition', 'stimulus']).fit()

    return {
        'table': result.anova_table,
        'dropped': dropped,
        'n_subjects': df['subject'].nunique(),
    }