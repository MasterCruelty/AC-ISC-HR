"""
single_recording.py --> Pipeline for one subject's ECG recording (one condition, one stimulus).

loading -> filtering -> peak detection -> instantaneous HR -> interpolation.

This is stage 1 of the project pipeline. 
It assembles the four processing modules to turn a single raw recording into one interpolated HR series.
Stage 2 (isc_analysis.py) calls this procedure once per subject to build the ISC-HR.
"""

from loading import load_ecg
from filtering import preprocess
from peak_detection import detect_r_peaks
from hr_analysis import instantaneous_hr, interpolate_hr


def process_single_recording(tsv_path, json_path, fs_common=4.0, verbose=False):
    """
    Process one ECG recording, from the raw signal to an HR series on a common time grid.

    Steps: load the recording, filter it (preprocess), detect the R peaks,
    compute the instantaneous HR from the R-R intervals and interpolate it
    onto a regular grid.

    Parameters
    ----------
    tsv_path : str
        Path to the ECG data file (.tsv.gz).
    json_path : str
        Path to the metadata file (.json), which provides the sampling
        frequency and the condition label.
    fs_common : float, optional
        Sampling frequency of the common grid, in Hz (default 4.0).
    verbose : bool, optional
        If True, print the number of detected peaks and the mean and standard
        deviation of the HR (default False).

    Returns
    -------
    t_common : numpy.ndarray
        Common time grid, in seconds.
    hr_interp : numpy.ndarray
        Instantaneous HR interpolated on t_common, in BPM.
    condition : str
        Condition label from the metadata, for example "Stim 01, Attentive Condition".
    n_peaks : int
        Number of detected R peaks.
    mean_hr : float
        Mean instantaneous HR in BPM, computed on the beat-by-beat values
        (before interpolation). It is used downstream for the quality control
        on the physiological HR range.
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
