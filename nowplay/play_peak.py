import sys
import time
import threading
import argparse
import sounddevice as sd
import numpy as np
from mutagen import File as MutagenFile
from .utils import load_audio

# Unicode blocks (unused here, but kept for reference)
BLOCKS = " ▁▂▃▄▅▆▇█"

# Default ANSI 256 colors for waveform (bottom to top)
DEFAULT_COLORS = [160, 166, 3, 46, 22][::-1]  # Reverse for top-to-bottom rendering
color_steps = DEFAULT_COLORS

def render_waveform_vertical(chunk, width=160, height=8):
    """
    Render an audio chunk as vertical waveform bars using ANSI colors.
    """
    if len(chunk) == 0:
        return ["   " * width for _ in range(height)]

    # Downsample to terminal width
    step = max(1, len(chunk) // width)
    sampled = np.abs(chunk[:step * width].reshape(-1, step).mean(axis=1))
    norm = sampled / np.max(sampled) if np.max(sampled) > 0 else sampled
    levels = (norm * (height - 1)).astype(int)

    rows = []
    # Build each row from top (height-1) down to 0
    for row in reversed(range(height)):
        line = ''
        for level in levels:
            # choose color index based on vertical position
            color_idx = (
                4 if row >= height * 0.9 else
                3 if row >= height * 0.7 else
                2 if row >= height * 0.5 else
                1 if row >= 2 else
                0
            )
            color = color_steps[color_idx]
            if level >= row:
                line += f"\033[38;5;{color}m███"
            else:
                line += '   '
        line += '\033[0m'  # reset at line end
        rows.append(line)
    return rows


def print_metadata(filepath):
    meta = MutagenFile(filepath, easy=True)
    if meta is None:
        print("No metadata found.")
        return
    print("Metadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")


def play_and_visualize(filepath):
    # Load audio
    y, sr = load_audio(filepath, mono=True)
    # Clear screen
    print("\033[H\033[J", end="")

    # Validate sample rate
    try:
        sd.check_output_settings(samplerate=sr, channels=1)
    except sd.PortAudioError:
        print(f"Warning: Sample rate {sr} not supported. Falling back to 44100 Hz.")
        sr = 44100
        y, _ = load_audio(filepath, sr=sr, mono=True)

    if y.size == 0:
        print("Failed to load audio.")
        return

    # Show metadata and duration
    print_metadata(filepath)
    duration = len(y) / sr
    print(f"Duration: {duration:.2f} seconds\n")

    blocksize = 1024
    cursor = [0]

    def callback(outdata, frames, time_info, status):
        idx = cursor[0]
        chunk = y[idx:idx+frames]
        if len(chunk) < frames:
            outdata[:len(chunk), 0] = chunk
            outdata[len(chunk):, 0] = 0
            raise sd.CallbackStop()
        outdata[:, 0] = chunk
        cursor[0] += frames

    stream = sd.OutputStream(samplerate=sr, channels=1,
                              callback=callback, blocksize=blocksize)

    with stream:
        while cursor[0] < len(y):
            elapsed = cursor[0] / sr
            end = min(cursor[0] + sr // 10, len(y))
            chunk = y[cursor[0]:end]
            bars = render_waveform_vertical(chunk, width=60, height=18)

            # move cursor up to overwrite previous waveform
            print("\033[H" * (18 + 2), end="")
            for line in bars:
                print(f"│{line}")
            print(f"└ Elapsed: {elapsed:.2f}s / {duration:.2f}s".ljust(80))

    print("\nPlayback finished.")


def main():
    parser = argparse.ArgumentParser(description="Play audio with waveform visualization")
    parser.add_argument("filepath", help="Audio file to play and visualize")
    parser.add_argument("--colors", metavar="CODES",
                        help="Comma-separated ANSI 256 color codes bottom-to-top")
    args = parser.parse_args()

    global color_steps
    if args.colors:
        try:
            codes = [int(c) for c in args.colors.split(",")]
            if len(codes) < 2:
                raise ValueError
            color_steps = codes
        except ValueError:
            print("Error: --colors requires at least two comma-separated integers, e.g. 160,166,3,46,22")
            sys.exit(1)

    try:
        play_and_visualize(args.filepath)
    except KeyboardInterrupt:
        print("\nExiting...")
        print("\033[H\033[J", end="")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
