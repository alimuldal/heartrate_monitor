import numpy as np
from scipy import signal

try:
    from pyfftw.interfaces import scipy_fftpack as fftpack
    fft_kw = {'threads':-1}
except ImportError:
    from scipy import fftpack
    fft_kw = {}

from matplotlib import pyplot as plt

import cv2
import cv2.cv as cv

from skimage import transform


def gaussian_downsample(frames, pyramid_levels=4):
    nt = frames.shape[0]
    for ii, frame in enumerate(frames):
        pyr = transform.pyramid_gaussian(frame.astype(np.float))
        for jj in xrange(pyramid_levels + 1):
            ds = pyr.next()
        if ii == 0:
            out = np.empty((nt,) + ds.shape, dtype=np.float)
        out[ii] = ds
    return out

def gaussian_downsample2(frames, pyramid_levels=4):
    nt = frames.shape[0]
    for ii, frame in enumerate(frames):
        ds = frame.astype(np.float)
        for jj in xrange(pyramid_levels):
            ds = cv2.pyrDown(ds)
        if ii == 0:
            out = np.empty((nt,) + ds.shape, dtype=np.float)
        out[ii] = ds
    return out


def laplacian_downsample(frames, pyramid_levels=4):
    nt = frames.shape[0]
    for ii, frame in enumerate(frames):
        pyr = transform.pyramid_laplacian(frame.astype(np.float))
        for jj in xrange(pyramid_levels + 1):
            ds = pyr.next()
        if ii == 0:
            out = np.empty((nt,) + ds.shape, dtype=np.float)
        out[ii] = ds
    return out


def laplacian_downsample2(frames, pyramid_levels=4):
    nt = frames.shape[0]
    for ii, frame in enumerate(frames):
        ds = frame.astype(np.float)
        for jj in xrange(pyramid_levels + 1):
            prev = ds.copy()
            ds = cv2.pyrDown(ds)
        laplacian = prev - cv2.pyrUp(ds)
        if ii == 0:
            out = np.empty((nt,) + laplacian.shape, dtype=np.float)
        out[ii] = laplacian
    return out

def butter_bandpass(frames, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')

    return signal.filtfilt(b, a, frames, axis=0)


def get_heartbeat_movie(frames, fps=60., bpm_limits=(40, 200),
                        pyramid_type='laplacian', pyramid_nlevels=4):

    nt, nr, nc = frames.shape
    frames = np.array(frames, dtype=np.float, copy=True)
    frames -= frames.min()
    frames /= frames.max()
    if pyramid_type == 'laplacian':
        downsamp = laplacian_downsample(frames, pyramid_nlevels)
    elif pyramid_type == 'gaussian':
        downsamp = gaussian_downsample(frames, pyramid_nlevels)

    lowcut, highcut = (ll / 60. for ll in bpm_limits)
    bandpassed = butter_bandpass(downsamp, lowcut, highcut, fps)

    # for ii, us in enumerate(bandpassed):
    #     for jj in xrange(pyramid_nlevels):
    #         us = cv2.pyrUp(us)
    #     frames[ii] = us[:nr, :nc]
    for ii, us in enumerate(bandpassed):
        for jj in xrange(pyramid_nlevels):
            us = transform.pyramid_expand(us)
        frames[ii] = us[:nr, :nc]

    return frames


def next_pow2(x, round_up=True):
    log2_x = np.log2(x)
    if round_up:
        return 2 ** int(np.ceil(log2_x))
    else:
        return 2 ** int(log2_x)


def get_heartrate(frames, fps, bpm_limits=(40, 200), min_window_sec=10,
                  plot=True):

    nyquist = 0.5 * fps
    lowcut, highcut = (ll / 60. for ll in bpm_limits)

    if lowcut > nyquist or highcut > nyquist:
        raise ValueError(
            'Filter critical frequencies must be <= Nyquist frequency')

    nt = frames.shape[0]
    pxsum = frames.reshape(nt, -1).sum(1)
    filt = butter_bandpass(pxsum.astype(np.float), lowcut, highcut, fps)
    detrended = signal.detrend(filt, type='linear')

    win = next_pow2(min_window_sec * fps)
    freq, psd = signal.welch(detrended, fps, nperseg=win,
                             return_onesided=True, scaling='density')

    valid = np.logical_and(freq >= lowcut, freq <= highcut)
    peak_idx = np.argmax(psd[valid])
    peak_hz = freq[valid][peak_idx]
    peak_power = psd[valid][peak_idx]
    heartrate = peak_hz * 60.

    if plot:

        fig = plt.figure()
        gs = plt.GridSpec(3, 1)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2])

        t = np.arange(nt) / fps

        ax1.plot(t, pxsum)
        ax1.set_ylabel('Integrated fluorescence')
        ax1.tick_params(labelbottom=False)
        ax1.grid(True, axis='x')

        ax2.plot(t, detrended)
        ax2.set_ylabel('Filtered and detrended')
        ax2.set_xlabel('Time (s)')
        ax2.grid(True, axis='x')
        ax2.set_xlim(0, t[-1])

        ax3.hold(True)
        ax3.axvspan(lowcut, highcut, color='r', alpha=0.1)
        ax3.semilogy(freq, psd)
        arrowprops=dict(arrowstyle='simple', fc='r', ec='None')
        ax3.annotate('%.2f bpm' % heartrate,
                     (peak_hz, peak_power), (0, -60),
                     xycoords='data', textcoords='offset points',
                     arrowprops=arrowprops, color='r', fontsize='x-large',
                     ha='center', va='bottom')
        ax3.set_ylabel('Power spectral density')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_xlim(0, highcut * 3.)
        ax3.set_ylim(psd[freq < highcut * 3].min(), None)

        plt.show()
    return heartrate