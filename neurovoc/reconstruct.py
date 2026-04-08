"""Module containing methods to reconstruct .wav files from Neurograms"""

import os
import pathlib
from math import gcd

from loguru import logger
import librosa
import scipy
import numpy as np

from .generate import Neurogram, min_max_scale


def rms(x):
    return np.sqrt(np.mean(x**2))


def rms_db(y):
    return 20 * np.log10(rms(y))


def scale_to_target_dbfs(y, target_dbfs):
    current_dbfs = rms_db(y)
    diff = target_dbfs - current_dbfs
    gain = 10 ** (diff / 20)
    return y * gain


def reconstruct_neurogram(
    M: np.ndarray, sr: int, min_freq: int, max_freq: int, n_fft: int, n_hop: int
) -> np.ndarray:
    """
    Parameters
    ----------
    M: np.ndarray
        A neurogram structure, scaled to a power spectrum, and downsampled by a factor
        of n_hop
    sr: int
        The sampling rate of the original neurogram (before resampling with n_hop)
    min_freq: int
        The lower bound of the filter bank
    max_freq: int
        The upper bound of the filter bank
    n_hop: int
        The number of hops that were applied to M
    """

    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=M.shape[-2],
        dtype=M.dtype,
        fmin=min_freq,
        fmax=max_freq,
    )
    inverse = librosa.util.nnls(mel_basis, M)
    inverse = np.power(inverse, 1.0 / 2.0, out=inverse)

    reconstructed = librosa.feature.inverse.griffinlim(
        inverse,
        n_iter=32,
        hop_length=n_hop,
        win_length=None,
        n_fft=n_fft,
        window="hann",
        center=True,
        dtype=np.float32,
        length=None,
        pad_mode="constant",
        momentum=0.99,
        init="random",
        random_state=None,
    )
    return reconstructed


def downsample(data: np.ndarray, n_hop: int) -> np.ndarray:
    n_s = int(np.ceil(data.shape[1] / n_hop))
    g = gcd(n_s, data.shape[1])
    data = np.array(
        [scipy.signal.resample_poly(row, n_s // g, data.shape[1] // g) for row in data]
    ).clip(0, 1)
    return data


def power_scale(data, ref_db: float = 50.0):
    data = min_max_scale(data, -80, 0, data_min=0, data_max=1)
    data = librosa.db_to_power(data, ref=ref_db)
    return data


def reconstruct(
    neurogram: Neurogram | str | pathlib.Path,
    n_hop: int = 32,
    n_fft: int = 512,
    ref_db: float = 50,
    target_sr: int = 44100,
    target_db_fs: int = -20,
    **kwargs,
):
    if isinstance(neurogram, (str, pathlib.Path)) and os.path.isfile(neurogram):
        logger.info(f"loading neurogram from file: {neurogram}")
        neurogram = Neurogram.load(neurogram)

    logger.info("downsample neurogram")
    data = downsample(neurogram.data, n_hop)

    logger.info("map to power scale")
    data = power_scale(data, ref_db)

    logger.info("reconstruct using griffin-lim")
    reconstructed = reconstruct_neurogram(
        data,
        neurogram.sample_rate,
        neurogram.min_freq,
        neurogram.max_freq,
        n_fft,
        n_hop,
    )
    logger.info("resample to original sample rate")
    reconstructed = librosa.resample(
        reconstructed, orig_sr=neurogram.sample_rate, target_sr=target_sr
    )
    logger.info("rescale to target dbfs")
    reconstructed = scale_to_target_dbfs(reconstructed, target_db_fs)
    return reconstructed
