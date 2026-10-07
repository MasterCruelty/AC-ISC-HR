"""
filtering.py --> Cleaning the raw ECG before peak detection.

"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def highpass_filter(x, fs, cutoff=0.5, order=4):
    """
    Remove the ECG signal using a highpass filter.

    Parameters
    ----------
    x : numpy.ndarray
        Raw ECG signal, one-dimensional array of samples.
    fs : float
        Sampling frequency of the signal, in Hz.
    cutoff : float, optional
        Cutoff frequency in Hz (default 0.5). Components slower than this are
        attenuated, while the faster ones (the heartbeats) are preserved.
    order : int, optional
        Order of the Butterworth filter (default 4).

    Returns
    -------
    numpy.ndarray
        Filtered signal, with the same length as x.
    """

    # Nyquist frequency
    nyq = fs / 2
    # calculating filter coefficients (b: numerator, a: denominator)
    b, a = butter(order, cutoff / nyq, btype='high')
    # applying filter defined by a and b to signal x.
    return filtfilt(b, a, x)


def notch_filter(x, fs, freq=60, Q=30):
    """
    Attenuate power-line interference at `freq` Hz.

    No-op if `freq` is at or above the Nyquist frequency (fs/2): at fs=128 Hz
    the Nyquist limit is 64 Hz, so a 60 Hz notch is technically applicable
    but very close to the edge. The effect is expected to be marginal in this case.

    Parameters
    ----------
    x : numpy.ndarray
        ECG signal (already highpass-filtered), one-dimensional array of samples.
    fs : float
        Sampling frequency of the signal, in Hz.
    freq : float, optional
        Frequency to suppress, in Hz (default 60, the power-line frequency
        of the recording site). Must be lower than fs / 2, otherwise the filter is not applied.
    Q : float, optional
        Quality factor (default 30): ratio between the notch frequency and
        the width of the suppressed band. With Q=30 at 60 Hz the band is
        about 2 Hz wide, so the rest of the signal is left almost untouched.

    Returns
    -------
    numpy.ndarray
        Filtered signal, with the same length as x. The filter is applied
        forward and backward (filtfilt), so the output has no phase delay.
        If freq >= fs / 2, the input x is returned unchanged.

    """
    nyq = fs / 2
    if freq >= nyq:
        return x
    b, a = iirnotch(freq / nyq, Q)
    return filtfilt(b, a, x)


def preprocess(raw, fs, notch_freq=60):
    """
    high-pass followed by notch.
    
    
    Parameters
    ----------
    raw : numpy.ndarray
        Raw ECG signal, one-dimensional array of samples.
    fs : float
        Sampling frequency of the signal, in Hz.
    notch_freq : float, optional
        Power-line frequency to attenuate, in Hz (default 60). If it is at or
        above fs / 2 the notch step is skipped.

    Returns
    -------
    numpy.ndarray
        Filtered ECG signal, with the same length as raw. Both filters are
        applied forward and backward (filtfilt), so the output has no phase
        delay.    
    """
    return notch_filter(highpass_filter(raw, fs), fs, freq=notch_freq)
