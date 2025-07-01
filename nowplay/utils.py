import numpy as np
import subprocess

def load_audio_ffmpeg(filepath, sr=None, mono=True, dtype=np.float32):
    """
    Load an audio file as a numpy array using ffmpeg and numpy.

    Args:
        filepath (str): Path to the audio file.
        sr (int or None): Target sampling rate. If None, uses the file's original.
        mono (bool): Convert to mono.
        dtype: Numpy dtype for output array.

    Returns:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of y.
    """
    try:
        # Probe audio info
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=channels,sample_rate",
            "-of", "default=noprint_wrappers=1:nokey=1", filepath
        ]
        probe = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        channels, orig_sr = probe.stdout.strip().split('\n')
        channels = int(channels)
        orig_sr = int(orig_sr)

        target_sr = sr if sr is not None else orig_sr
        target_channels = 1 if mono else channels

        ffmpeg_cmd = [
            "ffmpeg", "-i", filepath, "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ar", str(target_sr),
            "-ac", str(target_channels),
            "-"
        ]
        proc = subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        audio = np.frombuffer(proc.stdout, dtype=np.float32)

        if mono:
            y = audio
        else:
            y = audio.reshape(-1, target_channels)

        if dtype != np.float32:
            y = y.astype(dtype)
        return y, target_sr
    except FileNotFoundError as e:
        # Fallback: try to load raw PCM WAV files (very limited)
        try:
            with open(filepath, "rb") as f:
                header = f.read(44)
                if header[0:4] != b'RIFF' or header[8:12] != b'WAVE':
                    raise RuntimeError("File is not a standard PCM WAV file and ffmpeg is not available.")
                # Extract sample rate and channels from header
                orig_sr = int.from_bytes(header[24:28], "little")
                channels = int.from_bytes(header[22:24], "little")
                bits_per_sample = int.from_bytes(header[34:36], "little")
                if bits_per_sample != 16:
                    raise RuntimeError("Only 16-bit PCM WAV supported in fallback.")
                raw = f.read()
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if channels > 1:
                    audio = audio.reshape(-1, channels)
                if mono and channels > 1:
                    audio = np.mean(audio, axis=1)
                if dtype != np.float32:
                    audio = audio.astype(dtype)
                return audio, orig_sr
        except Exception as fallback_e:
            raise RuntimeError(
                "ffmpeg/ffprobe not found and fallback WAV loader failed: " + str(fallback_e)
            ) from fallback_e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg or ffprobe failed: {e.stderr}"
        ) from e

def load_audio(filepath, sr=None, mono=True, dtype=np.float32):
    """
    Load an audio file as a numpy array using ffmpeg and numpy.
    Falls back to a basic WAV loader if ffmpeg is not available.

    Args:
        filepath (str): Path to the audio file.
        sr (int or None): Target sampling rate. If None, uses the file's original.
        mono (bool): Convert to mono.
        dtype: Numpy dtype for output array.

    Returns:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of y.
    """
    try:
        return load_audio_ffmpeg(filepath, sr=sr, mono=mono, dtype=dtype)
    except RuntimeError as e:
        print(f"Warning: {e}")
        print("Falling back to dummy loader (returns empty array).")
        return np.array([], dtype=dtype), 0

