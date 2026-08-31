"""
single_recording.py — Pipeline for one recording (one subject, one condition,
one stimulus).

loading -> filtering -> peak detection -> instantaneous HR -> interpolation.

This is stage 1 of the pipeline: it assembles the four processing modules to
turn a single raw recording into one interpolated HR series.
Stage 2 (isc_analysis.py) calls this once per subject to build the ISC-HR.
"""

from loading import load_ecg
from filtering import preprocess
from peak_detection import detect_r_peaks
from hr_analysis import instantaneous_hr, interpolate_hr


def process_single_recording(tsv_path, json_path, fs_common=4.0, verbose=False):
    """
    

    Returns
    -------
    t_common : np.ndarray   — common time grid (s)
    hr_interp : np.ndarray  — interpolated instantaneous HR (BPM)
    condition : str         — for example "Stim 01, Attentive Condition"
    n_peaks : int           — number of detected R peaks 
    mean_hr : float         — mean instantaneous HR, BPM 
    """
    raw, fs, condition = load_ecg(tsv_path, json_path)
    filtered = preprocess(raw, fs)
    peaks, _ = detect_r_peaks(filtered, fs)
    t_hr, hr, rr = instantaneous_hr(peaks, fs)
    t_common, hr_interp = interpolate_hr(t_hr, hr, fs_common=fs_common)

    if verbose:
        print(f"  {condition}: {len(peaks)} peaks, "
              f"mean HR = {hr.mean():.1f} BPM (std {hr.std():.1f})")

    return t_common, hr_interp, condition, len(peaks), hr.mean()
