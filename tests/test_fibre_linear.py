"""Tests for the fibre_linear reconstruction (fibre rates -> linear STFT grid, no mel stage)."""

import os
import unittest

import librosa
import numpy as np

import neurovoc
from neurovoc import Neurogram, reconstruct
from neurovoc.neurogram import mel_scale
from neurovoc.reconstruct import fibres_to_linear_power, kernel_sigma_mel

BINSIZE = 3.6e-05
SR = 1 / BINSIZE
N_FFT = 2048
FFT_F = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
SIGMA = kernel_sigma_mel(150, 10_500, 64)  # the default for a 64-band neurogram


def output_peak_hz(audio, sr):
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    return np.fft.rfftfreq(len(audio), 1 / sr)[np.argmax(spec)]


class TestFibresToLinearPower(unittest.TestCase):
    def test_active_fibre_peaks_at_its_frequency(self):
        cf = np.geomspace(150, 10_500, 300)
        frames = np.zeros((len(cf), 4))
        k = np.argmin(np.abs(cf - 1000))
        frames[k] = 1.0
        S = fibres_to_linear_power(frames, cf, SR, N_FFT, sigma_mel=SIGMA, min_freq=150, max_freq=10_500)
        self.assertLessEqual(abs(FFT_F[np.argmax(S[:, 0])] - cf[k]), SR / N_FFT)

    def test_spectrum_is_dense_inside_the_band_and_zero_outside(self):
        cf = np.geomspace(150, 10_500, 300)
        frames = np.random.default_rng(0).random((len(cf), 4))
        S = fibres_to_linear_power(frames, cf, SR, N_FFT, sigma_mel=SIGMA, min_freq=150, max_freq=10_500)
        band = (FFT_F >= 150) & (FFT_F <= 10_500)
        self.assertTrue(np.all(S[band] > 0))
        self.assertTrue(np.all(S[~band] == 0))

    def test_fibre_density_does_not_bias_level(self):
        # a stretch with many fibres must not come out louder than one with few
        cf = np.r_[np.geomspace(300, 1000, 20), np.geomspace(2000, 6000, 200)]
        frames = np.full((len(cf), 2), 0.5)
        frames[0, 0] = 0.0  # one quiet fibre so min-max scaling has a range
        S = fibres_to_linear_power(frames, cf, SR, N_FFT, sigma_mel=SIGMA, min_freq=150, max_freq=10_500)
        lo = S[np.argmin(np.abs(FFT_F - 600)), 1]
        hi = S[np.argmin(np.abs(FFT_F - 4000)), 1]
        self.assertAlmostEqual(10 * np.log10(lo / hi), 0.0, delta=0.5)


class TestDownsample(unittest.TestCase):
    def test_matches_row_by_row_resampling(self):
        from math import gcd

        import scipy.signal

        from neurovoc.reconstruct import downsample

        x = np.random.default_rng(0).random((5, 10_007))
        n_s = int(np.ceil(x.shape[1] / 32))
        g = gcd(n_s, x.shape[1])
        expected = np.array(
            [scipy.signal.resample_poly(r, n_s // g, x.shape[1] // g) for r in x]
        ).clip(0, 1)
        np.testing.assert_allclose(downsample(x, 32), expected, atol=1e-12)


class TestReconstructMethods(unittest.TestCase):
    def setUp(self):
        self.freqs = mel_scale(64, 150, 10_500)
        data = np.zeros((64, int(0.5 / BINSIZE)))
        data[3] = 1.0
        self.ng = Neurogram(BINSIZE, self.freqs, data, "test")

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            reconstruct(self.ng, method="no-such-method")

    def test_default_is_fibre_linear(self):
        kw = dict(n_fft=N_FFT, target_sr=44_100, seed=1)  # Griffin-Lim starts from random phase
        default = reconstruct(self.ng, **kw)
        fibre_linear = reconstruct(self.ng, method="fibre_linear", **kw)
        mel = reconstruct(self.ng, method="mel", **kw)
        np.testing.assert_allclose(default, fibre_linear)
        self.assertFalse(np.allclose(default, mel))

    def test_both_methods_place_a_row_at_its_frequency(self):
        for method in ("fibre_linear", "mel"):
            peak = output_peak_hz(reconstruct(self.ng, n_fft=N_FFT, target_sr=44_100, method=method), 44_100)
            self.assertLess(abs(np.log2(peak / self.freqs[3])), 0.05, f"{method}: {peak:.0f} Hz")


class TestFibreLevelNeurogram(unittest.TestCase):
    """Runs a short phast simulation (a few seconds)."""

    def test_specres_fiber_level_keeps_every_fibre(self):
        root = os.path.dirname(os.path.dirname(__file__))
        ng = neurovoc.specres(os.path.join(root, "data", "025.wav"), n_trials=1, fiber_level=True)
        self.assertEqual(ng.data.shape[0], 64 * 10)
        self.assertTrue(np.all(np.diff(ng.frequencies) >= 0))  # rows sorted by frequency
        audio = reconstruct(ng)
        self.assertTrue(np.all(np.isfinite(audio)))


if __name__ == "__main__":
    unittest.main()
