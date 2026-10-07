"""
hr_analysis.py --> From R-peak positions to instantaneous heart-rate series.


1. instantanous_hr
At this point we have a indexes where R-peak were detected. They dont tell us
how fast the heart is beating, they tell us only "when" the heart has beaten.
To obtain the cardiac frequency there's need to obtain distance between consecutives beats.
Basically if the time betwwen two beat is short, the heart is beating fast. Otherwise it is slow.
If RR is the range of time between two consecutives R-peaks, the instantaneous heart rate in BPM is:
HR = 60 / RR

That's because if heart beat every RR seconds, during a minute it will beat 60/RR times.
Example: RR = 0.8 -> HR = 60 / 0.8 = 75 BPM


2. interpolate_hr
The hr series calculate with the previous function isnt sampled at regular intervals.
We have a HR value at every beat, but the beats aren't happening at stable intervals.
So we need two vectors of the same length aligned on the same temporal istants.

Solution: re-sample every individual HR serie on a common temporal grid by interpolation.
Interpolation estimates the value of a signal at instants where it was not measured, using the nearby measured points.


"""

import numpy as np
from scipy.interpolate import interp1d


def instantaneous_hr(peaks, fs):
    """
    Convert R-peak sample indices into an instantaneous HR series.

    Parameters:
    ----------
    peaks: np.ndarray
        Sample indices of the detected R peaks.
    fs: int
        sample frequency
    
    
    Returns
    ---------
    t_hr : np.ndarray
        timestamps (s), irregular
    hr   : np.ndarray
        instantaneous HR (BPM), irregular
    rr   : np.ndarray
        RR intervals (s)
    """

    # np.diff calculates difference between consecutives elements.
    # these are distances expressed in samples, not seconds.
    # Result: array of RR intervals expressed in seconds.
    rr_intervals = np.diff(peaks) / fs
    
    hr = 60 / rr_intervals

    #conversion of peak indexes in temporal instants in seconds.
    t_peaks = peaks / fs

    # t_peaks[:-1] are every peak times except last one.
    #t_peaks[1:] are every peak times except first.
    # by summing them and divide by 2, we obtain the average point for every couple.
    # that's the timestamp for that HR value.
    t_hr = (t_peaks[:-1] + t_peaks[1:]) / 2
    return t_hr, hr, rr_intervals


def interpolate_hr(t_hr, hr, fs_common=4.0, kind='cubic'):
    """
    Resample the irregular beat-to-beat HR series onto a regular grid.

    fs_common=4.0 Hz is a standard choice in the HRV literature.


    Parameters
    ----------
    t_hr : numpy.ndarray
        Timestamps of the HR values, in seconds, in increasing order
        (one per beat, at the midpoint of each R-R interval).
    hr : numpy.ndarray
        Instantaneous heart rate values, in BPM, same length as t_hr.
    fs_common : float, optional
        Sampling frequency of the common grid, in Hz (default 4.0, one
        sample every 0.25 s). It is not specified by the original paper, so it is an
        implementation choice.
    kind : str, optional
        Interpolation method passed to scipy.interpolate.interp1d
        (default 'cubic', a cubic spline).

    Returns
    -------
    t_common : numpy.ndarray
        Regular time grid, in seconds, from the first timestamp up to (but
        excluding) the last one.
    hr_interp : numpy.ndarray
        Estimated HR in BPM at every point of t_common.
    """
    # creation of the temporal grid, it starts from first timestamp to the last one.
    # the step is 1/4 = 0.25 seconds by giving fs_common = 4.0
    t_common = np.arange(t_hr[0], t_hr[-1], 1 / fs_common)

    # build of interpolation function that given irregolar points. 
    # it returns an estimated HR value for every average temporal istant requested, by using cubic specified method.
    f = interp1d(t_hr, hr, kind=kind)

    # Evaluation of estimated HR value for every point of the new grid t_common.
    # Result: definitive HR serie now at 0.25 seconds intervals.
    # It's ready to be compared with every subject re-sampled with the same procedure.
    hr_interp = f(t_common)
    return t_common, hr_interp
