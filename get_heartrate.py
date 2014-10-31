#!/usr/bin/env python

import os
import optparse
import numpy as np
from scipy import signal

import matplotlib
matplotlib.use('qt4agg')
from matplotlib import pyplot as plt

from sciscan_io import SciScanStack


def compute_heartrate(frames, fps, bpm_limits=(40, 200), window_sec=10,
                      do_plot=True):

    nyquist = 0.5 * fps
    lowcut, highcut = (ll / 60. for ll in bpm_limits)

    if lowcut > nyquist or highcut > nyquist:
        raise ValueError(
            'Filter critical frequencies must be <= Nyquist frequency')

    nt = frames.shape[0]
    pxsum = frames.reshape(nt, -1).sum(1).astype(np.float)
    detrended = signal.detrend(pxsum, type='linear')

    win = int(window_sec * fps)
    freq, psd = signal.welch(detrended, fps, nperseg=win, nfft=next_pow2(win),
                             return_onesided=True, scaling='density')

    valid = np.logical_and(freq >= lowcut, freq <= highcut)
    peak_idx = np.argmax(psd[valid])
    peak_hz = freq[valid][peak_idx]
    peak_power = psd[valid][peak_idx]
    heartrate = peak_hz * 60.

    print "-" * 50
    print "Estimated heartrate (bpm):\t%.3f" % heartrate
    print "Peak power spectral density:\t%.3g" % peak_power

    if do_plot:

        fig, (ax1, ax2) = plt.subplots(2, 1)
        t = np.arange(nt) / fps

        ax1.plot(t, pxsum)
        ax1.set_xlim(0, t[-1])
        ax1.set_ylabel('Integrated fluorescence')
        ax1.set_xlabel('Time (sec)')
        ax1.grid(True)

        ax2.hold(True)
        ax2.axvspan(lowcut, highcut, color='r', alpha=0.25)
        ax2.axvline(peak_hz, lw=2, c='r')
        ax2.semilogy(freq, psd)
        bbox_props = dict(boxstyle="round", fc="w", ec="r", lw=2)
        # arrowprops=dict(arrowstyle='simple', fc='r', ec='None')
        ax2.annotate('%.2f bpm' % heartrate,
                     xy=(1, 1), xytext=(-20, -20),
                     xycoords='axes fraction', textcoords='offset points',
                     color='r', fontsize='x-large',
                     # arrowprops=arrowprops,
                     bbox=bbox_props,
                     ha='right', va='top')
        ax2.set_ylabel('Power spectral density')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_xlim(0, highcut * 3.)
        ax2.set_ylim(psd[freq < highcut * 3].min(), None)
        ax2.grid(True)

        plt.show()

    return heartrate, peak_power


def butter_bandpass(frames, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')

    return signal.filtfilt(b, a, frames, axis=0)


def next_pow2(x, round_up=True):
    log2_x = np.log2(x)
    if round_up:
        return 2 ** int(np.ceil(log2_x))
    else:
        return 2 ** int(log2_x)


if __name__ == '__main__':

    parser = optparse.OptionParser()
    parser.add_option(
        '-c', '--channel-idx', dest='chan_idx', default=0,
        help="index of channel to use for multichannel stacks",
        metavar='IDX', type='int'
    )
    parser.add_option('', '--psd-window', action='store',
                      dest='window_sec', default=10,
                      help="the window used to compute the average PSD (sec)",
                      metavar='WINDOW', type='float'
                      )
    parser.add_option(
        '', '--low-cutoff', dest='bpm_low', default=40,
        help="minimum expected heartrate (bpm)",
        metavar='LOW', type='float'
    )
    parser.add_option(
        '', '--high-cutoff', dest='bpm_high', default=240,
        help="maximum expected heartrate (bpm)",
        metavar='HIGH', type='float'
    )
    parser.add_option(
        '-n', '--max-nframes', dest='max_nframes', default=-1,
        help="maximum number of frames to use", metavar='MAX', type=int
    )
    parser.add_option(
        '-d', '--disable-plot', action='store_false', dest='do_plot', default=1,
        help="disable plotting", metavar='PLOT'
    )
    options, args = parser.parse_args()

    if len(args) != 1:
        parser.error('One positional argument required (path to .raw file)')

    path = os.path.expanduser(args[0])

    if (not os.path.exists(path)) or (not os.path.splitext(path)[-1] == '.raw'):
        parser.error('First argument must be a path to a valid .raw file')

    dirpath = os.path.split(path)[0]
    stack = SciScanStack(dirpath)

    if list(stack.dim_names) == ['T', 'Y', 'X']:
        frames = stack.frames[:options.max_nframes]
    elif list(stack.dim_names) == ['T', 'C', 'Y', 'X']:
        frames = stack.frames[:options.max_nframes, options.chan_idx]
    else:
        parser.error('Unsupported experiment type: %s'
                     % (''.join(stack.dim_names)))

    fps = stack.metadata.frames_p_sec

    print "-" * 50
    print "Experiment name:\t%s" % stack.metadata.experiment_name
    print "Sample rate (Hz):\t%.3f" % fps
    print "Number of frames:\t%i" % frames.shape[0]

    heartrate, peak_pow = compute_heartrate(
        frames, fps, bpm_limits=(options.bpm_low, options.bpm_high),
        window_sec=options.window_sec, do_plot=options.do_plot)
