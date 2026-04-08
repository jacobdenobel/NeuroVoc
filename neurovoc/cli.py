import time
import functools

import click
import librosa
import scipy
import soundfile as sf
from loguru import logger
import matplotlib.pyplot as plt

from neurovoc import Neurogram, bruce, specres, reconstruct, audio_vs_reconstructed, ace


required_options = [
    click.argument("source", type=click.Path(exists=True)),
    click.argument("target", type=click.Path(), required=False),
    click.option("--ref-db", type=float, default=50, help="Reference dB SPL"),
]
generate_options = [
    click.option("--n-trials", type=int, default=20, help="Number of trials per fiber"),
    click.option("--min-freq", type=int, default=150),
    click.option("--max-freq", type=int, default=10500),
    click.option("--n-mels", type=int, default=64),
    click.option("--n-fibers-per-bin", type=int, default=10),
    click.option("--window-size", type=int, default=1500),
    click.option("--normalize/--no-normalize", default=True),
    click.option("--seed", type=int, default=42),
    click.option("--n-threads", type=int, default=-1),
    click.option("--binsize", type=float, default=3.6e-05),
]

bruce_options = [
    click.option("--n-rep", type=int, default=1),
    click.option("--remove-outliers/--keep-outliers", default=True),
]

phast_options = [
    click.option("--spont-rate", type=int, default=50),
    click.option("--accommodation-amplitude", type=float, default=0.07),
    click.option("--adaptation-amplitude", type=float, default=7.142),
    click.option("--accommodation-rate", type=float, default=2.0),
    click.option("--adaptation-rate", type=float, default=19.996),
]

specres_options = [
    click.option("--current-steering/--no-current-steering", default=True),
] + phast_options

ace_options = [
    click.option("--version", type=click.Choice(["18_0", "25_8"]), default="25_8")
] + phast_options

reconstruct_options = [
    click.option("--n-hop", type=int, default=32, help="Hop length for STFT/ISTFT"),
    click.option("--n-fft", type=int, default=512, help="FFT window size"),
    click.option(
        "--target-sr",
        type=int,
        default=44100,
        help="Target sampling rate for output audio",
    ),
    click.option(
        "--target-db-fs",
        type=int,
        default=-20,
        help="Target dB FS for waveform normalization",
    ),
]


def apply_options(options):
    def wrapper(f):
        for option in reversed(options):
            f = option(f)
        return f

    return wrapper


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"⏱️ Time elapsed: {time.time() - start:.2f} seconds")
        return result

    return wrapper


def save_ng(neurogram: Neurogram, audio_path: str, save_to: str = None):
    if save_to is None:
        save_to = audio_path.replace(".wav", f"_{neurogram.source}_neurogram.pkl")
    neurogram.save(save_to)
    logger.info(f"Saved neurogram at: {save_to}")
    return save_to


def save_wav(reconstructed, source, target, sr, label=""):
    if target is None:
        if "neurogram" in source:
            target = source.replace("_neurogram.pkl", f"_{label}_reconstructed.wav")
        else:
            target = source.replace(".wav", f"_{label}_reconstructed.wav")

    sf.write(target, reconstructed, sr, subtype="PCM_32")
    logger.info(f"Saved reconstruction to: {target}")


def reconstruct_plot_save(source: str, target: str, neurogram: Neurogram, **kwargs):
    audio_signal, audio_fs = librosa.load(source, sr=None)
    kwargs["target_sr"] = audio_fs
    reconstructed_signal = reconstruct(neurogram, **kwargs)
    if reconstructed_signal.size != audio_signal.size:
        reconstructed_signal = scipy.signal.resample(
            reconstructed_signal, audio_signal.size
        )

    logger.info(f"Finished reconstructing signal: {reconstructed_signal.shape}")
    save_wav(
        reconstructed_signal,
        source,
        target,
        kwargs["target_sr"],
        label=neurogram.source,
    )
    if kwargs["plot"]:
        audio_vs_reconstructed(
            audio_signal,
            reconstructed_signal,
            audio_fs,
            len(neurogram.frequencies),
            neurogram.min_freq,
            neurogram.max_freq,
        )
        plt.show()
    return reconstructed_signal


@click.group()
def main():
    """NeuroVoc CLI"""
    pass


@main.group()
def generate():
    """Generate a neurogram."""
    pass


@main.group()
def vocode():
    """Run the complete vocoder (generate + reconstruct)"""
    pass


@main.command("reconstruct")
@apply_options(required_options + reconstruct_options)
@timeit
def reconstruct_cmd(source, target, **kwargs):
    """
    Reconstruct a waveform from a neurogram representation.

    Arguments:
      source  Path to neurogram file (.pkl)
      target  (Optional) Output file path (.wav)
    """
    reconstructed = reconstruct(source, **kwargs)
    logger.info(f"Finished reconstructing signal: {reconstructed.shape}")
    save_wav(reconstructed, source, target, kwargs["target_sr"])


@generate.command("bruce")
@apply_options(required_options + generate_options + bruce_options)
@timeit
def generate_bruce(source, target, **kwargs):
    """
    Generate a neurogram using the Bruce et al. NH model.

    Arguments:
      source  Path to input audio file (.wav or .npy)
      target  (Optional) Output file path to save neurogram (.npz)
    """
    neurogram = bruce(source, **kwargs)
    logger.info(f"Simulation completed. Generated {neurogram}")
    save_ng(neurogram, source, target)


@generate.command("specres")
@apply_options(required_options + generate_options + specres_options)
@timeit
def generate_specres(source, target, **kwargs):
    """
    Generate a neurogram using the Spectral Resolution 120 CI model.

    Arguments:
      source  Path to input audio file (.wav or .npy)
      target  (Optional) Output file path to save neurogram (.npz)
    """
    neurogram = specres(source, **kwargs)
    logger.info(f"Simulation completed. Generated {neurogram}")
    save_ng(neurogram, source, target)


@generate.command("ace")
@apply_options(required_options + generate_options + ace_options)
@timeit
def generate_ace(source, target, **kwargs):
    """
    Generate a neurogram using the Advanced Combinatorial Encoder CI model.

    Arguments:
      source  Path to input audio file (.wav or .npy)
      target  (Optional) Output file path to save neurogram (.npz)
    """
    neurogram = ace(source, **kwargs)
    logger.info(f"Simulation completed. Generated {neurogram}")
    save_ng(neurogram, source, target)


@vocode.command("specres")
@apply_options(
    required_options + generate_options + specres_options + reconstruct_options
)
@click.option("--plot/--no-plot", default=True)
@timeit
def vocode_specres(source, target, **kwargs):
    """
    Run the vocoder using the Spectral Resolution 120 CI model.

    Arguments:
      source  Path to input audio file (.wav or .npy)
      target  (Optional) Output file path (.wav)
    """

    neurogram = specres(source, **kwargs)
    logger.info(f"Simulation completed. Generated {neurogram}")
    reconstruct_plot_save(source, target, neurogram, **kwargs)


@vocode.command("bruce")
@apply_options(
    required_options + generate_options + bruce_options + reconstruct_options
)
@click.option("--plot/--no-plot", default=True)
@timeit
def vocode_bruce(source, target, **kwargs):
    """
    Run the vocoder using the Bruce et al. NH model.

    Arguments:
      source  Path to input audio file (.wav or .npy)
      target  (Optional) Output file path (.wav)
    """

    neurogram = bruce(source, **kwargs)
    logger.info(f"Simulation completed. Generated {neurogram}")
    reconstruct_plot_save(source, target, neurogram, **kwargs)


@vocode.command("ace")
@apply_options(required_options + generate_options + ace_options + reconstruct_options)
@click.option("--plot/--no-plot", default=True)
@timeit
def vocode_ace(source, target, **kwargs):
    """
    Run the vocoder using the  Advanced Combinatorial Encoder CI model.

    Arguments:
      source  Path to input audio file (.wav or .npy)
      target  (Optional) Output file path (.wav)
    """

    neurogram = ace(source, **kwargs)
    logger.info(f"Simulation completed. Generated {neurogram}")
    reconstruct_plot_save(source, target, neurogram, **kwargs)


if __name__ == "__main__":
    pass
