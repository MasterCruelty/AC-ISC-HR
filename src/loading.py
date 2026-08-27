"""
loading.py — Reading raw ECG recordings and their BIDS metadata.

"""
import json
import glob
import os
import pandas as pd


def load_ecg(tsv_path, json_path):
    """
    Load a single raw ECG recording (mV) plus its metadata.

    Parameters
    ----------
    tsv_path : str
        Path to the .tsv.gz file (single column 'rawECG', no header).
    json_path : str
        Path to the matching .json  (sampling frequency, condition).

    Returns
    -------
    ecg['rawECG'].values: np.ndarray
        Raw ECG samples, in mV.
    fs : float
        Sampling frequency, in Hz.
    condition : str
        Value of the 'TaskName' field (for example "Stim 01, Attentive Condition").
    """
    with open(json_path) as f:
        meta = json.load(f)
    ecg = pd.read_csv(tsv_path, sep='\t', header=None, names=['rawECG'], compression='gzip')
    fs = meta['SamplingFrequency']
    condition = meta['TaskName']
    return ecg['rawECG'].values, fs, condition


def list_subjects(experiment_root):
    """
    Discover subject IDs actually present on dataset.

    Subject numbering in BBBD is not contiguous (for example sub-04 may be missing),
    
    Returns
    -------
    list[str] : sorted subject folder names. Example ['sub-01', 'sub-02', ...]
    """
    pattern = os.path.join(experiment_root, 'sub-*')
    return sorted(os.path.basename(p) for p in glob.glob(pattern) if os.path.isdir(p))


def ecg_paths(experiment_root, subject, session, stim):
    """
    Build the (tsv, json) path pair for a given subject/session/stimulus,
    following the BBBD BIDS naming convention.

    session : str, for example 'ses-01' (Attentive) or 'ses-02' (Distracted)
    stim    : str, for example 'stim01'
    """
    base = os.path.join(
        experiment_root, subject, session, 'beh',
        f'{subject}_{session}_task-{stim}_recording-ecg_physio'
    )
    return base + '.tsv.gz', base + '.json'
