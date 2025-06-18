import sys
import time
import threading
import sounddevice as sd
import numpy as np
from mutagen import File as MutagenFile
from .utils import load_audio

BLOCKS = " ▁▂▃▄▅▆▇█"  # Unicode blocks from low to high

def render_waveform_vertical(chunk, width=160, height=8):
    if len(chunk) == 0:
        return [" " * width for _ in range(height)]

    # Downsample to terminal width
    step = max(1, len(chunk) // width)
    sampled = np.abs(chunk[:step * width].reshape(-1, step).mean(axis=1))
    norm = sampled / np.max(sampled) if np.max(sampled) > 0 else sampled
    levels = (norm * (height - 1)).astype(int)

    # Generate vertical bars (top-down lines)
    rows = []
    for row in reversed(range(height)):
        line = ''.join(
            f'\033[38;5;{160 if row > height/1.2 else 166 if row > height/1.7 else 3 if row > height/3 else 46 if row > 1 else 22}m███' if level >= row else '   '
            for level in levels
        )
        rows.append(line + '\033[0m')  # Reset color at end of line
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
    y, sr = load_audio(filepath, mono=True)
    print("\033[H\033[J", end="")
    # Validate sample rate for sounddevice
    try:
        sd.check_output_settings(samplerate=sr, channels=1)
    except sd.PortAudioError as e:
        print(f"Warning: Sample rate {sr} not supported. Falling back to 44100 Hz.")
        sr = 44100
        y, _ = load_audio(filepath, sr=sr, mono=True)

    if y.size == 0:
        print("Failed to load audio.")
        return

    print_metadata(filepath)
    duration = len(y) / sr
    print(f"Duration: {duration:.2f} seconds\n")

    blocksize = 1024
    cursor = [0]
    start_time = time.time()

    def callback(outdata, frames, time_info, status):
        idx = cursor[0]
        chunk = y[idx:idx+frames]
        if len(chunk) < frames:
            outdata[:len(chunk), 0] = chunk
            outdata[len(chunk):, 0] = 0
            raise sd.CallbackStop()
        else:
            outdata[:, 0] = chunk
        cursor[0] += frames

    stream = sd.OutputStream(
        samplerate=sr, channels=1, callback=callback, blocksize=blocksize
    )
    meta_shown = False
    print("\033[H\033[J", end="")
    with stream:
        while cursor[0] < len(y):
            elapsed = cursor[0] / sr
            end = min(cursor[0] + sr // 10, len(y))  # 0.1s chunk for visual
            chunk = y[cursor[0]:end]
            bars = render_waveform_vertical(chunk, width=60, height=18)
            print("\033[H" * 9)  # Move cursor up 9 lines (8 waveform + 1 elapsed line)
            for line in bars:
                print(f"│{line}")
            print(f"└ Elapsed: {elapsed:.2f}s / {duration:.2f}s".ljust(80))
            if not meta_shown:
                desired_keys = ["title", "artist", "album", "copyright"]
                meta = MutagenFile(filepath, easy=True)
                if meta:
                    for key in desired_keys:
                        value = meta.get(key)
                        if value:
                            print(f"  {key}: {value[0]}")
                else:
                    print("  No metadata found.")
                meta_shown = True
                print("\033[H" * (len(meta))) # Move cursor back up.
            #time.sleep(0.07)
    print("\nPlayback finished.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python play.py <audiofile>")
        sys.exit(1)
    try:
        play_and_visualize(sys.argv[1])
    except KeyboardInterrupt:
        print("\033[H\033[J", end="")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()