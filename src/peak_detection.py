"""
peak_detection.py — Locating R peaks in the filtered ECG(electrocardiogram).

Here we find the temporal position of every R-peak.
approach: amplitude thresold + minimum distance.

1. Amplitude thresold:
I consider a peak a point that is over a certain height thresold. Which thresold?
Instead of setting an absolute Mv number(it would work only for that specific subject),
the thresold is calculated based on the variability of the signal itself.
This is made using the std of the filtered signal as unit of measurement.
If a subject has an ECG with a larger amplitude signal, the thresold increase proportionally. 
Otherwise it decrease.
The moltiplicator factor is 3, so assuming that an event over 3 std is rare for only noise and fluctuations.
So we assume that a value of that height is a R-peak and not a casual noise.

2. Minimum distance criterium:
Without it, the problem could be detecting two really near peak in the same beat instead of a single one.
The solutions is to set a minimum distance between two consecutive detections.
Assuming 200 BPM(beat per minute) the minimum possible range between two beat is 60/200 = 0.3 seconds.

"""

import numpy as np
from scipy.signal import find_peaks


def detect_r_peaks(filtered_ecg, fs, min_rr_s=0.3, height_std_mult=3.0):
    """
    Detect R peaks via an adaptive amplitude threshold plus a physiological
    refractory distance.

    Parameters
    ----------
    min_rr_s : float
        Minimum allowed interval between consecutive beats, in seconds.
        0.3 s corresponds to a ceiling of 200 BPM; it prevents double-detection.        
    height_std_mult : float
        Amplitude threshold, expressed as a multiple of the filtered
        signal's standard deviation.

    Returns
    -------
    peaks : np.ndarray
        Sample indices of the detected R peaks.
    threshold : float
        The amplitude threshold actually used (mV), for logging/QC.
    """


    #conversion of temporal contraint in sample number.
    min_distance_samples = int(min_rr_s * fs)
    #calculate thresold
    threshold = height_std_mult * np.std(filtered_ecg)
    #find all local max that satisfy height thresold and minimum distance contraint.
    #the output is an array of indexes where these peaks were found. 
    #The 2nd value returned would be a dictionary with more property not needed in this case.
    peaks, _ = find_peaks(filtered_ecg, height=threshold, distance=min_distance_samples)
    return peaks, threshold
