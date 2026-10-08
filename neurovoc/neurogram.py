import pickle
from dataclasses import dataclass, field
from fractions import Fraction

import librosa
import numpy as np
import scipy


@dataclass
class Neurogram:
    dt: float
    frequencies: np.ndarray = field(repr=None)
    data: np.ndarray = field(repr=None)
    source: str 
    shape: tuple = None
    # number of frequency bands the neurogram resolves (its mel rows); for a
    # fibre-level neurogram the rows are fibres, more than n_mels
    n_mels: int = None

    def __post_init__(self):
        self.shape = self.data.shape

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @property
    def duration(self):
        return self.data.shape[1] * self.dt
    
    @property
    def sample_rate(self) -> int:
        return int(np.ceil(1 / self.dt))
    
    @property
    def min_freq(self) -> float:
        return np.min(self.frequencies)
    
    @property
    def max_freq(self) -> float:
        return np.max(self.frequencies)

    @staticmethod
    def load(path: str) -> "Neurogram":
        with open(path, "rb") as f:
            return pickle.load(f)


def mel_scale(n_mels: int, min_freq: int, max_freq: int):
    return librosa.filters.mel_frequencies(n_mels, fmin=min_freq, fmax=max_freq)


def bin_edges(frequencies: np.ndarray) -> np.ndarray:
    """Edges (Hz) of the bins centred on ``frequencies``, midway on the mel scale.

    The outer edges lie half a spacing beyond the first and last centre, so
    each row of a neurogram collects the fibres nearest to its own frequency.
    """
    mel = librosa.hz_to_mel(np.asarray(frequencies, dtype=float))
    mid = (mel[1:] + mel[:-1]) / 2
    edges = np.r_[mel[0] - (mel[1] - mel[0]) / 2, mid, mel[-1] + (mel[-1] - mel[-2]) / 2]
    return librosa.mel_to_hz(edges)


def bin_index(src_y: np.ndarray, tgt_y: np.ndarray) -> np.ndarray:
    """Row of ``tgt_y`` nearest (on the mel scale) to each ``src_y``; -1 outside all bins."""
    edges = bin_edges(tgt_y)
    idx = np.digitize(src_y, edges) - 1
    idx[(idx < 0) | (idx >= len(tgt_y))] = -1
    return idx


def bin_over_y(
    data: np.ndarray, src_y: np.ndarray, tgt_y: np.ndarray, agg: callable = np.sum
):
    """Aggregate the rows of ``data`` (one per ``src_y``) into one row per ``tgt_y``.

    Each source row goes to the target row whose frequency is nearest on the
    mel scale (``bin_edges``), so a row labelled f(i) is centred on f(i).
    Sources beyond the outer edges are dropped.
    """
    data_binned = np.zeros((len(tgt_y), data.shape[1]))
    bins = bin_index(np.asarray(src_y), tgt_y)

    for i in range(len(tgt_y)):
        if not any(bins == i):
            continue
        data_binned[i] = agg(data[bins == i], axis=0)
    return data_binned


def smooth(
    data: np.ndarray,
    window_type: str = "hann",
    window_size: int = 2048,
    hop_length: int = None,
) -> np.ndarray:
    """Hann (or other window) smoothing along time, centred on each sample."""
    hop_length = hop_length or max(window_size // 4, 1)
    window = scipy.signal.get_window(window_type, window_size)
    window /= window.sum()
    data = scipy.signal.oaconvolve(
        np.asarray(data, dtype=float), window[None, :], mode="same", axes=1
    )
    return data[:, ::hop_length]


def min_max_scale(
    data: np.ndarray,
    a: float = -80,
    b: float = 0,
    data_min: float = None,
    data_max: float = None,
):
    data_min = np.min(data) if data_min is None else data_min
    data_max = np.max(data) if data_max is None else data_max
    return a + (data - data_min) * (b - a) / (data_max - data_min)


def remove_outliers(data: np.ndarray, quantile: float = 0.995) -> np.ndarray:
    """Clip ``data`` at its ``quantile`` and rescale to [0, 1]."""
    data = np.clip(data, 0, np.quantile(data.ravel(), quantile))
    return min_max_scale(data, 0, 1)


def rebin_signal(signal, orig_sr, target_sr):
    signal = np.asarray(signal)
    frac = Fraction(target_sr, orig_sr).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator

    upsampled = np.zeros(len(signal) * up)
    upsampled[::up] = signal

    n_bins = len(upsampled) // down
    rebinned = upsampled[: n_bins * down].reshape(n_bins, down).sum(axis=1)

    return rebinned


def rebin_data(data: np.ndarray, dt_data: float, dt_tgt: float):
    src_sr = int(round(1 / dt_data))
    tgt_sr = int(round(1 / dt_tgt))
    return np.vstack([rebin_signal(x, src_sr, tgt_sr) for x in data])


def make_bins(n, data):
    if n == 1:
        return data
    return data[:, : len(data[0]) // n * n].reshape(data.shape[0], -1, n).sum(axis=2)
