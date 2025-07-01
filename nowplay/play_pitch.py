import sys
import time
import argparse
import numpy as np
import sounddevice as sd
from mutagen import File as MutagenFile
from blessed import Terminal

from .utils import load_audio

fast = False
pretty = False

BLOCKS = " ▁▂▃▄▅▆▇█"  # Optional
DEFAULT_COLORS = [160, 166, 3, 46, 22]


def get_metadata(filepath):
    desired_keys = ["title", "artist", "album", "copyright"]
    meta = MutagenFile(filepath, easy=True)
    metadata = {}
    if meta:
        for key in desired_keys:
            value = meta.get(key)
            if value:
                metadata[key] = value[0]
    return metadata


def render_spectrum_bars(chunk, width=60, height=18, sr=44100, floor_db=-60.0, f_min=20.0, ref=1.0):
    if len(chunk) == 0:
        return [0] * width

    w     = np.hanning(len(chunk))
    spec  = np.abs(np.fft.rfft(chunk * w))
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)
    f_max = sr / 2
    edges = np.logspace(np.log10(f_min), np.log10(f_max), num=width + 1)

    bands = []
    for i in range(width):
        lo, hi = edges[i], edges[i + 1]
        idx    = np.where((freqs >= lo) & (freqs < hi))[0]
        bands.append(spec[idx].max() if idx.size else 0.0)

    mags_db = 20 * np.log10(np.array(bands) / (ref + 1e-9) + 1e-9)
    mags_db = np.clip(mags_db, floor_db, None)
    norm    = (mags_db - floor_db) / (-floor_db)
    levels  = (norm * height).astype(int)
    return levels


def play_and_visualize(filepath, color_steps_override=None):
    term = Terminal()
    y, sr = load_audio(filepath, mono=True)
    if y.size == 0:
        print(term.red("Failed to load audio."))
        return

    try:
        sd.check_output_settings(samplerate=sr, channels=1)
    except sd.PortAudioError:
        sr = 44100
        y, _ = load_audio(filepath, sr=sr, mono=True)

    # Determine color steps
    color_steps = color_steps_override if color_steps_override is not None else DEFAULT_COLORS

    window        = np.hanning(len(y))
    full_spectrum = np.abs(np.fft.rfft(y * window))
    file_peak     = np.percentile(full_spectrum, 90)
    duration      = len(y) / sr
    metadata      = get_metadata(filepath)

    blocksize = 1024
    def callback(outdata, frames, time_info, status):
        idx   = callback.idx
        chunk = y[idx:idx + frames]
        if len(chunk) < frames:
            outdata[:len(chunk), 0] = chunk
            outdata[len(chunk):, 0] = 0
            raise sd.CallbackStop()
        outdata[:, 0] = chunk
        callback.idx += frames
    callback.idx = 0

    hop    = sr // 10
    fps    = 60 if pretty else 15 if fast else 30
    frames = int(np.ceil(duration * fps))
    spectrum_height = 18
    spectrum_width  = 60

    with sd.OutputStream(samplerate=sr, channels=1, callback=callback, blocksize=blocksize):
        with term.fullscreen(), term.hidden_cursor():
            start_time = time.time()
            for frame in range(frames):
                elapsed = time.time() - start_time
                idx = int(elapsed * sr)
                if idx >= len(y):
                    break

                chunk = y[idx : idx + hop]
                levels = render_spectrum_bars(chunk, width=spectrum_width, height=spectrum_height, sr=sr, ref=file_peak)

                output_lines = []

                for row in range(spectrum_height):
                    line = ""
                    for col, lvl in enumerate(levels):
                        color_idx = (
                            4 if row >= spectrum_height * 0.9 else
                            3 if row >= spectrum_height * 0.7 else
                            2 if row >= spectrum_height * 0.5 else
                            1 if row >= spectrum_height * 0.3 else
                            0
                        )
                        color_code = color_steps[color_idx]
                        char = (
                            "█" if spectrum_height - row <= lvl and fast else
                            "███" if spectrum_height - row <= lvl and pretty else
                            "██" if spectrum_height - row <= lvl else
                            "".ljust(2) if fast else
                            "".ljust(3) if pretty else
                            "".ljust(2)
                        )
                        if char.strip():
                            line += term.color(color_code)(char)
                        else:
                            line += char
                    output_lines.append(term.move(row, 0) + line)

                output_lines.append(term.move(spectrum_height, 0) +
                                    f"└ Elapsed: {elapsed:.2f}s / {duration:.2f}s".ljust(80))

                for i, (key, value) in enumerate(metadata.items()):
                    output_lines.append(term.move(spectrum_height + 2 + i, 0) + f"{key}: {value}")

                full_frame = term.move(0, 0) + "".join(output_lines)
                sys.stdout.write(full_frame)
                sys.stdout.flush()

                time.sleep(max(0, (start_time + (frame + 1) / fps) - time.time()))

    print(term.move(spectrum_height + 7, 0) + term.green("Playback finished."))


def main():
    parser = argparse.ArgumentParser(description="Play audio with spectrum visualization")
    parser.add_argument("filepath", help="Path to the audio file to play and visualize")
    parser.add_argument("--fast", action="store_true", help="Use fast mode (smaller blocks)")
    parser.add_argument("--pretty", action="store_true", help="Use pretty mode (wider blocks)")
    parser.add_argument("--colors", type=str,
                        help="Comma-separated ANSI 256 color codes for the spectrum gradient (bottom to top)")
    args = parser.parse_args()

    global fast, pretty
    fast = args.fast
    pretty = args.pretty

    color_override = None
    if args.colors:
        try:
            codes = [int(c) for c in args.colors.split(",")]
            if len(codes) < 2:
                raise ValueError
            color_override = codes
        except ValueError:
            print("Error: --colors must be a comma-separated list of integers (e.g. 160,166,3,46,22)")
            sys.exit(1)

    try:
        play_and_visualize(args.filepath, color_steps_override=color_override)
    except KeyboardInterrupt:
        print("\nExiting...")
        print("\033[H\033[J", end="")  # Clear screen
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
