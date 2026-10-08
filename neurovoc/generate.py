"""Module to generate neurograms with phast"""

import pathlib
import numpy as np
import phast
import brucezilany
from loguru import logger

from .neurogram import (
    Neurogram,
    mel_scale,
    bin_index,
    bin_over_y,
    smooth,
    min_max_scale,
    rebin_data,
    remove_outliers as clip_outliers,
)


def get_electrode_freq_specres():
    return np.r_[
        phast.scs.ab.defaults.ELECTRODE_FREQ_LOWER,
        phast.scs.ab.defaults.ELECTRODE_FREQ_UPPER[-1],
    ]


def get_electrode_freq_ace():
    bin_freq_Hz = 15625 / 128
    band_bins = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8]
    freqs = []
    current_bin = 2  # Skip DC and first bin
    for width in band_bins:
        start_bin = current_bin
        end_bin = current_bin + width - 1
        center_bin = (start_bin + end_bin) / 2
        freqs.append((center_bin * bin_freq_Hz))
        current_bin += width
    return np.array(freqs)[::-1]


def _contact_fibre_distance(i_det: np.ndarray, tp: phast.ThresholdProfile) -> float:
    """Median distance (mm) between each contact and its lowest-threshold fibre."""
    first_fibre = np.asarray(tp.position)[np.argmin(i_det, axis=0)]
    return float(np.median(np.abs(first_fibre - np.asarray(tp.electrode.position))))


def load_cochlear_profile(version: str = "25_8") -> phast.ThresholdProfile:
    """``phast.load_cochlear`` with its threshold matrix in the same order as its labels.

    The stored data number electrodes the Cochlear way (E1 basal first) and
    fibres from base to apex. ``phast.load_cochlear`` (1.1.7) flips the
    electrode and fibre metadata (positions, T/M levels, Greenwood
    frequencies) to apical-first, but not ``i_det``: every contact then
    seems to excite fibres 4-6 mm away from it, and ACE drives its lowest
    band onto the most basal contact. Here ``i_det`` is flipped on both axes
    and the T/M levels are stored basal-first, so that ``ace_e2e``'s own
    flip hands each band the levels of the contact that now carries it.

    Only applied when needed: if ``i_det`` already agrees with the labels
    (each contact's lowest-threshold fibre near the contact), the profile
    is returned as loaded.
    """
    tp = phast.load_cochlear(version=version)
    i_det = np.asarray(tp.i_det)
    flipped = np.ascontiguousarray(i_det[::-1, ::-1])
    if _contact_fibre_distance(flipped, tp) >= _contact_fibre_distance(i_det, tp):
        return tp

    el = tp.electrode
    electrode = phast.ElectrodeConfiguration(
        m_level=np.flip(el.m_level),
        t_level=np.flip(el.t_level),
        insertion_angle=el.insertion_angle,
        greenwood_f=el.greenwood_f,
        position=el.position,
        pw=el.pw,
        ipg=el.ipg,
    )
    return phast.ThresholdProfile(
        i_det=flipped,
        electrode=electrode,
        angle=tp.angle,
        position=tp.position,
        greenwood_f=tp.greenwood_f,
        fiber_type=tp.fiber_type,
    )


def get_fiber_freq_position(
    tp: phast.ThresholdProfile, electrode_freq: np.array
) -> np.ndarray:
    """Frequency assigned to each fibre: the electrode frequencies interpolated
    along the cochlea, Greenwood's range at the ends.

    Contacts are paired with frequencies by position (most basal contact =
    highest frequency), whatever order either array is given in: the ACE
    frequencies come in Cochlear numbering (E1 basal first) while the
    threshold profile stores electrodes apical first.
    """
    position = np.asarray(tp.position, dtype=float)
    e_pos = np.sort(np.asarray(tp.electrode.position, dtype=float))  # base -> apex
    e_freq = np.sort(np.asarray(electrode_freq, dtype=float))[::-1]  # high -> low
    fiber_freq = np.interp(
        position,
        np.r_[position.min(), e_pos, position.max()],
        np.r_[tp.greenwood_f.max(), e_freq, tp.greenwood_f.min()],
    )
    return fiber_freq


def select_fibers(
    fiber_freq: np.ndarray, frequency_bins: np.ndarray, n_fibers_per_bin: int = 10
) -> np.ndarray:
    """``n_fibers_per_bin`` fibres for every frequency bin, drawn from the
    fibres nearest to that bin's frequency (the bins of ``bin_over_y``).
    Bins with fewer fibres get duplicates."""
    grouped = bin_index(np.asarray(fiber_freq), frequency_bins)
    selected_fibers = []
    for fbin, nf in zip(*np.unique(grouped[grouped >= 0], return_counts=True)):
        fibers = np.where(grouped == fbin)[0]
        if nf < n_fibers_per_bin:
            n_extra = n_fibers_per_bin - nf
            sf = np.r_[
                fibers, np.random.choice(fibers, n_extra, replace=n_extra > nf)
            ]
        else:
            sf = np.random.choice(fibers, n_fibers_per_bin, replace=False)
        selected_fibers.extend(sorted(sf))
    return np.array(selected_fibers, dtype=int)


def add_duplicate_fibers_to_tp(tp: phast.ThresholdProfile, selected_fibers: np.ndarray):
    uni, cnt = np.unique(selected_fibers, return_counts=True)
    new_fidx = []
    for fidx, n_copies in zip(uni[cnt > 1], cnt[cnt > 1] - 1):
        for _ in range(n_copies):
            tp.i_det = np.vstack([tp.i_det, tp.i_det[fidx, :]])
            tp.greenwood_f = np.r_[tp.greenwood_f, tp.greenwood_f[fidx]]
            tp.position = np.r_[tp.position, tp.position[fidx]]
            tp.angle = np.r_[tp.angle, tp.angle[fidx]]
            new_fidx.append(tp.i_det.shape[0] - 1)

    return np.r_[uni, new_fidx].astype(int)


def configure_fiberset(
    tp: phast.ThresholdProfile,
    electrode_freq: np.array,
    frequencies: np.array,
    n_fibers_per_bin: int = 10,
):
    fiber_freq = get_fiber_freq_position(tp, electrode_freq)
    selected_fibers = select_fibers(fiber_freq, frequencies, n_fibers_per_bin)
    # duplicates are appended to tp and listed after the unique fibres, so look the
    # frequencies up in the order phast will simulate (and return) the fibres
    selected_fibers = add_duplicate_fibers_to_tp(tp, selected_fibers)
    fiber_freq = get_fiber_freq_position(tp, electrode_freq)[selected_fibers]
    return selected_fibers, fiber_freq


def load_audio(
    audio: str | pathlib.Path | np.ndarray, audio_fs: int = None, ref_db: float = 50.0
):
    if isinstance(audio, (str, pathlib.Path)):
        audio_signal, audio_fs = phast.scs.ab.frontend.read_wav(audio, stim_db=ref_db)
    elif isinstance(audio, np.ndarray):
        audio_signal = phast.scs.ab.frontend.process_stim(
            audio, audio_fs, stim_db=ref_db
        )
        audio_fs = 17400
    else:
        if audio_fs is None:
            raise TypeError(
                "Wrong audio signal type. If a numpy array is given, audio_fs must also be passed"
            )
    audio_signal += np.random.normal(0, 1e-20, size=audio_signal.shape)
    duration = len(audio_signal.ravel()) * (1 / audio_fs)
    return audio_signal, audio_fs, duration


def process_neurogram(
    neurogram_data: np.ndarray,
    fiber_freq: np.ndarray,
    frequencies: np.ndarray,
    window_size: int,
    binsize: float,
    normalize: bool,
    name: str,
    duration: float,
    fiber_level: bool = False,
):
    n_mels = len(frequencies)
    if fiber_level:
        logger.info("keeping one row per fibre, sorted by frequency")
        order = np.argsort(fiber_freq, kind="stable")
        neurogram_data = np.asarray(neurogram_data)[order]
        frequencies = np.asarray(fiber_freq)[order]
    else:
        logger.info("rebin neurogram")
        neurogram_data = bin_over_y(neurogram_data, fiber_freq, frequencies, agg=np.sum)

    # The end of response does not mean the end of the stimulus
    ng_duration = neurogram_data.shape[1] * binsize
    missing = duration - ng_duration
    zero_columns = int(np.ceil(missing / binsize))
    if missing > 0:
        neurogram_data = np.c_[
            neurogram_data, np.zeros((neurogram_data.shape[0], zero_columns))
        ]
    elif missing < 0:
        neurogram_data = neurogram_data[:, :zero_columns]

    if window_size is not None:
        logger.info(f"applying hann filter of size {window_size}")
        neurogram_data = smooth(neurogram_data, "hann", window_size, 1)

    if normalize:
        logger.info("scaling neurogram to [0,1)")
        neurogram_data = min_max_scale(neurogram_data, 0, 1)

    return Neurogram(binsize, frequencies, neurogram_data, name, n_mels=n_mels)


def specres(
    audio: str | pathlib.Path | np.ndarray,
    audio_fs: int = None,
    ref_db: float = 50,
    n_trials: int = 20,
    current_steering: int = True,
    min_freq: int = 150,
    max_freq: int = 10_500,
    n_mels: int = 64,
    spont_rate: int = 50,
    n_fibers_per_bin: int = 10,
    accommodation_amplitude: float = 0.07,
    adaptation_amplitude: float = 7.142,
    accommodation_rate: float = 2.0,
    adaptation_rate: float = 19.996,
    window_size: int = 1500,
    normalize: bool = True,
    seed: int = 42,
    binsize: float = 3.6e-05,
    n_threads: int = -1,
    fiber_level: bool = False,
    **kwargs,
) -> Neurogram:
    if isinstance(audio, (str, pathlib.Path)):
        logger.info(f"loading audio from file: {audio}")
    else:
        logger.info(f"using audio signal of length {len(audio)} with sr {audio_fs}")
    phast.set_seed(seed)
    np.random.seed(seed)
    frequencies = mel_scale(n_mels, min_freq, max_freq)

    tp = phast.load_df120()

    assert max_freq > phast.scs.ab.defaults.ELECTRODE_FREQ_UPPER[-1]

    selected_fibers, fiber_freq = configure_fiberset(
        tp, get_electrode_freq_specres(), frequencies, n_fibers_per_bin
    )
    audio_signal, audio_fs, duration = load_audio(audio, audio_fs, ref_db)
    logger.info("creating neurogram")
    (audio_signal, FS), pulse_train, neurogram = phast.ab_e2e(
        audio_signal=audio_signal,
        audio_fs=audio_fs,
        tp=tp,
        current_steering=current_steering,
        scaling_factor=1.4,
        ramp_duration=(audio_signal.size / audio_fs) * 0.05,
        n_trials=n_trials,
        accommodation_amplitude=accommodation_amplitude,
        adaptation_amplitude=adaptation_amplitude,
        accommodation_rate=accommodation_rate,
        adaptation_rate=adaptation_rate,
        selected_fibers=selected_fibers,
        spont_activity=spont_rate,
        stim_db=ref_db,
        binsize=binsize,
        n_jobs=n_threads,
        **kwargs,
    )
    logger.info("processing neurogram")
    neurogram = process_neurogram(
        neurogram.data,
        fiber_freq,
        frequencies,
        window_size,
        binsize,
        normalize,
        "phast_specres",
        duration,
        fiber_level,
    )
    return neurogram


def ace(
    audio: str | pathlib.Path | np.ndarray,
    audio_fs: int = None,
    ref_db: int = 50,
    n_trials: int = 20,
    min_freq: int = 150,
    max_freq: int = 10_500,
    n_mels: int = 64,
    spont_rate: int = 50,
    n_fibers_per_bin: int = 10,
    accommodation_amplitude: float = 0.07,
    adaptation_amplitude: float = 7.142,
    accommodation_rate: float = 2.0,
    adaptation_rate: float = 19.996,
    window_size: int = 1500,
    normalize: bool = True,
    seed: int = 42,
    binsize: float = 3.6e-05,
    n_threads: int = -1,
    version: str = "25_8",
    fiber_level: bool = False,
    **kwargs,
) -> Neurogram:
    if isinstance(audio, (str, pathlib.Path)):
        logger.info(f"loading audio from file: {audio}")
    else:
        logger.info(f"using audio signal of length {len(audio)} with sr {audio_fs}")
    phast.set_seed(seed)
    np.random.seed(seed)
    frequencies = mel_scale(n_mels, min_freq, max_freq)
    tp = load_cochlear_profile(version=version)

    selected_fibers, fiber_freq = configure_fiberset(
        tp, get_electrode_freq_ace(), frequencies, n_fibers_per_bin
    )
    audio_signal, audio_fs, duration = load_audio(audio, audio_fs, ref_db)
    audio_signal = audio_signal.ravel()
    logger.info("creating neurogram")
    (audio_signal, FS), pulse_train, neurogram = phast.ace_e2e(
        audio_signal=audio_signal,
        audio_fs=audio_fs,
        tp=tp,
        n_trials=n_trials,
        accommodation_amplitude=accommodation_amplitude,
        adaptation_amplitude=adaptation_amplitude,
        accommodation_rate=accommodation_rate,
        adaptation_rate=adaptation_rate,
        selected_fibers=selected_fibers,
        spont_activity=spont_rate,
        stim_db=ref_db,
        n_jobs=n_threads,
        binsize=binsize,
        **kwargs,
    )
    logger.info("processing neurogram")
    neurogram = process_neurogram(
        neurogram.data,
        fiber_freq,
        frequencies,
        window_size,
        binsize,
        normalize,
        "phast_ace",
        duration,
        fiber_level,
    )

    return neurogram


def add_noise(audio, f1=0.05, f2=0.05):

    n1 = int(len(audio) * f1)
    n2 = int(len(audio) * f2)
    noise = np.random.uniform(
        low=0.8 * audio.max(), high=1.2 * audio.max(), size=n1
    ) * np.random.choice([-1, 1], size=n1)
    noise = np.r_[noise, np.random.normal(0, scale=1e-14, size=n2)]
    padded = np.r_[noise, audio]
    return padded, 1 - (len(audio) / len(padded))


def bruce(
    audio: str | pathlib.Path | np.ndarray,
    audio_fs: int = None,
    ref_db: int = 50,
    n_trials: int = 20,
    min_freq: int = 150,
    max_freq: int = 10_500,
    n_mels: int = 64,
    n_fibers_per_bin: int = 10,
    window_size: int = 1500,
    normalize: bool = True,
    seed: int = 42,
    n_threads: int = -1,
    binsize: float = 3.6e-05,
    n_rep: int = 1,
    remove_outliers: bool = True,
    **kwargs,
):
    if isinstance(audio, (str, pathlib.Path)):
        logger.info(f"loading audio from file: {audio}")
    else:
        logger.info(f"using audio signal of length {len(audio)} with sr {audio_fs}")

    brucezilany.set_seed(seed)
    np.random.seed(seed)

    if isinstance(audio, (str, pathlib.Path)):
        stim = brucezilany.stimulus.from_file(audio, False, normalize=False)
    elif isinstance(audio, np.ndarray):
        duration = (1 / audio_fs) * len(audio)
        stim = brucezilany.stimulus.Stimulus(audio, audio_fs, duration)
    else:
        if audio_fs is None:
            raise TypeError(
                "Wrong audio signal type. If a numpy array is given, audio_fs must also be passed"
            )

    stim = brucezilany.stimulus.normalize_db(stim, ref_db)

    frequencies = mel_scale(n_mels, min_freq, max_freq)

    n_low = n_med = int(np.floor(n_fibers_per_bin / 5))
    n_high = n_fibers_per_bin - (2 * n_med)
    logger.info("creating neurogram")
    ng = brucezilany.Neurogram(
        frequencies,
        n_low=n_low,
        n_med=n_med,
        n_high=n_high,
        n_threads=n_threads,
    )
    ng.bin_width = stim.time_resolution
    ng.create(stim, n_rep=n_rep, n_trials=n_trials)
    logger.info("processing neurogram")
    neurogram_data = ng.get_output().sum(axis=1)
    neurogram_data = rebin_data(neurogram_data, stim.time_resolution, binsize)

    if window_size is not None:
        logger.info(f"applying hann filter of size {window_size}")
        neurogram_data = smooth(neurogram_data, "hann", window_size, 1)

    if normalize:
        logger.info("scaling neurogram to [0,1)")
        neurogram_data = min_max_scale(neurogram_data, 0, 1)

    if remove_outliers:
        logger.info("removing outliers")  # TODO: why is this after normalization?
        neurogram_data = clip_outliers(neurogram_data, 0.995)

    return Neurogram(binsize, frequencies, neurogram_data, "brucezilany", n_mels=n_mels)
