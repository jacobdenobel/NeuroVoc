"""Regression tests for the binning / reconstruction fixes.

Each test pins one defect of v0.1.2 that shifted energy in frequency or time,
or distorted levels, between the fibre neurogram and the reconstructed audio.
All tests are synthetic and fast (no phast/bruce simulation).
"""

import unittest

import librosa
import numpy as np
import phast

from neurovoc import Neurogram, reconstruct
from neurovoc.generate import (
    get_electrode_freq_ace,
    get_electrode_freq_specres,
    get_fiber_freq_position,
    load_cochlear_profile,
    process_neurogram,
    select_fibers,
)
from neurovoc.neurogram import mel_scale, min_max_scale, remove_outliers, smooth
from neurovoc.reconstruct import invert_mel_power, mel_basis_for

BINSIZE = 3.6e-05
FREQS = mel_scale(64, 150, 10_500)


def nearest_row(freq_hz, frequencies=FREQS):
    """Index of the row whose centre is nearest on the mel scale."""
    d = np.abs(librosa.hz_to_mel(np.atleast_1d(freq_hz))[:, None] - librosa.hz_to_mel(frequencies)[None, :])
    return np.argmin(d, axis=1)


def output_peak_hz(audio, sr):
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    return np.fft.rfftfreq(len(audio), 1 / sr)[np.argmax(spec)]


class TestNeurogramHelpers(unittest.TestCase):
    def test_min_max_scale_accepts_zero_as_data_min(self):
        # data_min=0 used to be treated as "not given" (falsy) and replaced by min(data)
        scaled = min_max_scale(np.array([0.5, 1.0]), 0, 1, data_min=0, data_max=1)
        np.testing.assert_allclose(scaled, [0.5, 1.0])

    def test_smooth_is_centred(self):
        # the window used to look ahead (output[k] = mean of x[k:k+N]), shifting events early
        x = np.zeros((1, 2000))
        x[0, 1000] = 1.0
        y = smooth(x, "hann", 301, 1)
        self.assertLessEqual(abs(int(np.argmax(y[0])) - 1000), 1)

    def test_remove_outliers_clips_to_quantile(self):
        # bruce used to discard the result of clip(), so outliers were never removed
        data = np.linspace(0, 1, 10_000).reshape(10, 1000)
        data[0, 0] = 100.0
        out = remove_outliers(data, 0.995)
        self.assertAlmostEqual(out.max(), 1.0)
        self.assertGreater(np.quantile(out, 0.5), 0.45)  # bulk no longer squashed towards 0


class TestFrequencyBinning(unittest.TestCase):
    def test_fibre_at_row_frequency_lands_in_that_row(self):
        # bin_over_y used to put fibres from [f(i-1), f(i)) into row i: labels = upper edge
        fiber_freq = np.array([FREQS[10], FREQS[40]])
        data = np.array([[1.0, 1.0], [2.0, 2.0]])
        ng = process_neurogram(data, fiber_freq, FREQS, None, BINSIZE, False, "test", 2 * BINSIZE)
        self.assertEqual(ng.data[10, 0], 1.0)
        self.assertEqual(ng.data[40, 0], 2.0)

    def test_fibre_just_above_row_frequency_lands_in_that_row(self):
        mel = librosa.hz_to_mel(FREQS)
        step = mel[1] - mel[0]
        f = librosa.mel_to_hz(mel[20] + 0.3 * step)  # nearer to row 20 than to row 21
        ng = process_neurogram(np.ones((1, 2)), np.array([f]), FREQS, None, BINSIZE, False, "test", 2 * BINSIZE)
        self.assertEqual(np.flatnonzero(ng.data[:, 0]).tolist(), [20])

    def test_select_fibers_fills_each_row_with_its_nearest_fibres(self):
        np.random.seed(0)
        fiber_freq = np.geomspace(100, 12_000, 3000)[::-1]  # base (high) first, as phast
        selected = select_fibers(fiber_freq, FREQS, 10)
        rows = nearest_row(fiber_freq[selected])
        np.testing.assert_array_equal(np.bincount(rows, minlength=len(FREQS)), np.full(len(FREQS), 10))

    def test_ace_fibre_frequencies_decrease_from_base_to_apex(self):
        # get_electrode_freq_ace() is high->low but electrode positions are apical-first,
        # so the most apical contact was paired with the highest band (scrambled CFs)
        tp = phast.load_cochlear()
        cf = get_fiber_freq_position(tp, get_electrode_freq_ace())
        order = np.argsort(tp.position)  # distance from base, ascending
        self.assertTrue(np.all(np.diff(cf[order]) <= 1e-9))

    def test_cochlear_profile_contacts_excite_fibres_at_their_own_place(self):
        # phast.load_cochlear flips the electrode and fibre labels to apical-first but not
        # i_det, so each contact's lowest-threshold fibre appeared 4-6 mm away from it
        tp = load_cochlear_profile()
        first_fibre = np.asarray(tp.position)[np.argmin(np.asarray(tp.i_det), axis=0)]
        distance = np.abs(first_fibre - np.asarray(tp.electrode.position))
        self.assertLess(np.median(distance), 1.5)

    def test_specres_fibre_frequencies_decrease_from_base_to_apex(self):
        tp = phast.load_df120()
        cf = get_fiber_freq_position(tp, get_electrode_freq_specres())
        order = np.argsort(tp.position)
        self.assertTrue(np.all(np.diff(cf[order]) <= 1e-9))


class TestMelInversion(unittest.TestCase):
    def test_mel_filters_peak_at_row_frequencies(self):
        # librosa.filters.mel(n, fmin, fmax) centres its filters on the inner points of an
        # n + 2 grid, not on mel_frequencies(n, fmin, fmax): up to one channel off at the ends
        sr, n_fft = 1 / BINSIZE, 8192
        basis = mel_basis_for(FREQS, sr, n_fft)
        peaks = librosa.fft_frequencies(sr=sr, n_fft=n_fft)[np.argmax(basis, axis=1)]
        bin_hz = sr / n_fft
        np.testing.assert_allclose(peaks, FREQS, atol=bin_hz)

    def test_inversion_reproduces_low_level_rows(self):
        # librosa.util.nnls did not converge on the floor of an 80 dB-range target
        sr, n_fft = 1 / BINSIZE, 1024
        basis = mel_basis_for(FREQS, sr, n_fft)
        db = np.full(len(FREQS), -80.0)
        db[[10, 30, 50]] = 0.0
        mel_power = np.repeat(librosa.db_to_power(db)[:, None], 3, axis=1)
        S = invert_mel_power(basis, mel_power, FREQS, sr, n_fft)
        err_db = 10 * np.log10((basis @ S)[:, 0] / mel_power[:, 0])
        ok = basis.sum(axis=1) > 0
        self.assertLess(np.median(np.abs(err_db[ok])), 1.0)
        self.assertTrue(np.all(S >= 0))

    def test_single_row_reconstructs_at_its_frequency(self):
        for row in (3, 32, 60):
            data = np.zeros((len(FREQS), int(0.5 / BINSIZE)))
            data[row] = 1.0
            ng = Neurogram(BINSIZE, FREQS, data, "test")
            audio = reconstruct(ng, n_fft=2048, target_sr=44_100)  # 13.6 Hz bins: resolves row 3
            peak = output_peak_hz(audio, 44_100)
            self.assertLess(abs(np.log2(peak / FREQS[row])), 0.05, f"row {row}: {peak:.0f} Hz vs {FREQS[row]:.0f} Hz")


if __name__ == "__main__":
    unittest.main()
