"""
filtering.py — Cleaning the raw ECG before peak detection.

"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def highpass_filter(x, fs, cutoff=0.5, order=4):
    """
    Detrend the ECG signal using a highpass filter.
    """

    # Nyquist frequency
    nyq = fs / 2
    # calculating math coefficients of highpass filter
    b, a = butter(order, cutoff / nyq, btype='high')
    # applying filter defined by a and b to signal x.
    return filtfilt(b, a, x)


def notch_filter(x, fs, freq=60, Q=30):
    """
    Remove power-line interference at `freq` Hz.

    No-op if `freq` is at or above the Nyquist frequency (fs/2): at fs=128 Hz
    the Nyquist limit is 64 Hz, so a 60 Hz notch is technically applicable
    but very close to the edge — its effect is expected to be marginal in this case.
    """
    nyq = fs / 2
    if freq >= nyq:
        return x
    b, a = iirnotch(freq / nyq, Q)
    return filtfilt(b, a, x)


def preprocess(raw, fs, notch_freq=60):
    """high-pass followed by notch."""
    return notch_filter(highpass_filter(raw, fs), fs, freq=notch_freq)
