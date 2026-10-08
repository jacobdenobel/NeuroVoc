"""Module containing methods to reconstruct .wav files from Neurograms"""

import os
import pathlib
from math import gcd

from loguru import logger
import librosa
import scipy
import numpy as np

from .generate import Neurogram, min_max_scale
from .neurogram import mel_scale


def rms(x):
    return np.sqrt(np.mean(x**2))


def rms_db(y):
    return 20 * np.log10(rms(y))


def scale_to_target_dbfs(y, target_dbfs):
    current_dbfs = rms_db(y)
    diff = target_dbfs - current_dbfs
    gain = 10 ** (diff / 20)
    return y * gain


def mel_basis_for(frequencies: np.ndarray, sr: float, n_fft: int) -> np.ndarray:
    """Mel filterbank with one filter centred on each neurogram row frequency.

    ``librosa.filters.mel(n, fmin, fmax)`` centres its filters on the inner
    points of an ``n + 2`` grid, whereas the rows are labelled
    ``mel_frequencies(n, fmin, fmax)`` (endpoints included). Extending fmin
    and fmax by one mel step makes the two coincide.
    """
    mel = librosa.hz_to_mel(np.asarray(frequencies, dtype=float))
    step = np.diff(mel)
    if not np.allclose(step, step.mean(), rtol=1e-6):
        raise ValueError("neurogram rows are not uniformly spaced on the mel scale")
    fmin = float(librosa.mel_to_hz(mel[0] - step.mean()))
    fmax = float(librosa.mel_to_hz(mel[-1] + step.mean()))
    if fmax >= sr / 2:
        raise ValueError(f"top filter edge {fmax:.0f} Hz exceeds Nyquist ({sr / 2:.0f} Hz)")
    return librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=len(mel), fmin=fmin, fmax=fmax
    )


def invert_mel_power(
    mel_basis: np.ndarray,
    mel_power: np.ndarray,
    frequencies: np.ndarray,
    sr: float,
    n_fft: int,
    n_iter: int = 50,
) -> np.ndarray:
    """Dense, non-negative linear power spectrogram S with ``mel_basis @ S ~ mel_power``.

    ``librosa.util.nnls`` does not converge on the low-level parts of an
    80 dB-range target and returns a sparse spectrum (most bins exactly zero).
    Here the start is the per-bin power density of each row, interpolated
    along the mel axis in the log domain; it is refined with ``n_iter``
    Richardson-Lucy (KL) multiplicative updates, S <- S * A^T(M / AS) / A^T 1,
    which weigh low-level rows by their relative error and keep S positive.
    """
    A = np.asarray(mel_basis, dtype=float)
    M = np.asarray(mel_power, dtype=float)
    fft_f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    rowsum = A.sum(axis=1)
    ok = rowsum > 0
    if ok.sum() < 2:
        raise ValueError("mel basis has fewer than 2 non-empty filters; increase n_fft")

    tiny = 1e-12 * M.max() if M.max() > 0 else 1e-30
    log_density = np.log(M[ok] / rowsum[ok, None] + tiny)
    x_mel = librosa.hz_to_mel(np.asarray(frequencies, dtype=float))[ok]
    q_mel = librosa.hz_to_mel(fft_f)
    S = np.exp(np.stack([np.interp(q_mel, x_mel, col) for col in log_density.T], axis=1))

    colsum = A.sum(axis=0)
    support = colsum > 0
    S[~support] = 0.0  # outside the filterbank
    for _ in range(n_iter):
        ratio = M / (A @ S + tiny)
        S[support] *= (A.T @ ratio)[support] / colsum[support, None]
    return S


def reconstruct_neurogram(
    M: np.ndarray,
    sr: int,
    min_freq: int,
    max_freq: int,
    n_fft: int,
    n_hop: int,
    frequencies: np.ndarray = None,
    seed: int = None,
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
    frequencies: np.ndarray, optional
        The row frequencies of M; by default mel_scale(n_rows, min_freq, max_freq)
    seed: int, optional
        Seed for Griffin-Lim's random initial phase (None: not reproducible)
    """
    if frequencies is None:
        frequencies = mel_scale(M.shape[-2], min_freq, max_freq)
    mel_basis = mel_basis_for(frequencies, sr, n_fft)
    inverse = invert_mel_power(mel_basis, M, frequencies, sr, n_fft)
    return griffinlim(np.sqrt(inverse), n_fft, n_hop, seed)


def griffinlim(magnitude: np.ndarray, n_fft: int, n_hop: int, seed: int = None) -> np.ndarray:
    """Waveform from a linear magnitude spectrogram (fast Griffin-Lim, 32 iterations)."""
    reconstructed = librosa.feature.inverse.griffinlim(
        magnitude,
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
        random_state=seed,
    )
    return reconstructed


def downsample(data: np.ndarray, n_hop: int) -> np.ndarray:
    n_s = int(np.ceil(data.shape[1] / n_hop))
    g = gcd(n_s, data.shape[1])
    # one call for all rows: the filter is designed once (it is as long as the
    # signal for most lengths, so designing it per row took seconds per row)
    data = scipy.signal.resample_poly(
        data, n_s // g, data.shape[1] // g, axis=1
    ).clip(0, 1)
    return data


def power_scale(data, ref_db: float = 50.0):
    data = min_max_scale(data, -80, 0, data_min=0, data_max=1)
    data = librosa.db_to_power(data, ref=ref_db)
    return data


def fibres_to_linear_power(
    frames: np.ndarray,
    fibre_freq: np.ndarray,
    sr: float,
    n_fft: int,
    sigma_mel: float,
    min_freq: float,
    max_freq: float,
    ref_db: float = 50.0,
) -> np.ndarray:
    """Fibre rates (fibres x frames) -> linear power spectrogram (fft bins x frames).

    Each STFT bin gets the Gaussian-kernel-weighted mean rate of the fibres
    whose frequency lies near it on the mel scale (Nadaraya-Watson, so a
    stretch with many fibres is not louder than one with few). There is no
    mel filterbank to invert. The rates are min-max scaled and mapped to
    power as in ``power_scale``; bins outside [min_freq, max_freq], or with
    no fibre within ~4 sigma, are zero.
    """
    frames = np.asarray(frames, dtype=float)
    fft_f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = np.flatnonzero((fft_f >= min_freq) & (fft_f <= max_freq))
    d = (
        librosa.hz_to_mel(fft_f[band])[:, None]
        - librosa.hz_to_mel(np.asarray(fibre_freq, dtype=float))[None, :]
    ) / sigma_mel
    w = np.exp(-0.5 * d**2)
    w[np.abs(d) > 4] = 0.0
    wsum = w.sum(axis=1)
    covered = wsum > 1e-3
    rates = (w[covered] @ frames) / wsum[covered, None]

    S = np.zeros((len(fft_f), frames.shape[1]))
    S[band[covered]] = power_scale(min_max_scale(rates, 0, 1), ref_db)
    return S


def kernel_sigma_mel(min_freq: float, max_freq: float, n_bands: int) -> float:
    """Kernel width: half the spacing of ``n_bands`` mel bands over [min_freq, max_freq]."""
    span = librosa.hz_to_mel(max_freq) - librosa.hz_to_mel(min_freq)
    return 0.5 * float(span) / (n_bands - 1)


METHODS = ("fibre_linear", "mel")


def reconstruct(
    neurogram: Neurogram | str | pathlib.Path,
    n_hop: int = 32,
    n_fft: int = 512,
    ref_db: float = 50,
    target_sr: int = 44100,
    target_db_fs: int = -20,
    method: str = "fibre_linear",
    seed: int = None,
    **kwargs,
):
    """Audio from a neurogram.

    method:
        ``"fibre_linear"`` (default): every row is taken as a fibre at its own
        frequency and the rates are kernel-averaged straight onto the linear
        STFT grid (``fibres_to_linear_power``) - no mel stage. Works on mel-binned
        neurograms and on fibre-level ones (``specres``/``ace`` with
        ``fiber_level=True``).
        ``"mel"``: the rows are mel bands; the mel filterbank is inverted
        (``invert_mel_power``). Needs rows uniformly spaced on the mel scale.
    seed:
        Seed for Griffin-Lim's random initial phase (default None, as before).
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if isinstance(neurogram, (str, pathlib.Path)) and os.path.isfile(neurogram):
        logger.info(f"loading neurogram from file: {neurogram}")
        neurogram = Neurogram.load(neurogram)

    logger.info("downsample neurogram")
    data = downsample(neurogram.data, n_hop)

    if method == "fibre_linear":
        n_bands = getattr(neurogram, "n_mels", None) or data.shape[0]
        sigma = kernel_sigma_mel(neurogram.min_freq, neurogram.max_freq, n_bands)
        logger.info("map fibre rates onto the linear frequency grid")
        power = fibres_to_linear_power(
            data,
            neurogram.frequencies,
            neurogram.sample_rate,
            n_fft,
            sigma,
            neurogram.min_freq,
            neurogram.max_freq,
            ref_db,
        )
        logger.info("reconstruct using griffin-lim")
        reconstructed = griffinlim(np.sqrt(power), n_fft, n_hop, seed)
    else:
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
            frequencies=neurogram.frequencies,
            seed=seed,
        )
    logger.info("resample to original sample rate")
    reconstructed = librosa.resample(
        reconstructed, orig_sr=neurogram.sample_rate, target_sr=target_sr
    )
    logger.info("rescale to target dbfs")
    reconstructed = scale_to_target_dbfs(reconstructed, target_db_fs)
    return reconstructed
