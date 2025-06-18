#!/usr/bin/env python3
import sys
import time

import sounddevice as sd
import numpy as np
from mutagen import File as MutagenFile

from .utils import load_audio

BLOCKS = " ▁▂▃▄▅▆▇█"  # Unicode blocks from low to high

def render_spectrum_vertical(chunk,
                             width=160, height=8,
                             sr=44100,
                             floor_db=-60.0,
                             f_min=20.0,
                             ref=1.0):
    """
    Log-spaced FFT → peak band energies → dB relative to `ref` → floor/normalize → ANSI bars.
    """
    if len(chunk) == 0:
        return ['   ' * width for _ in range(height)]

    # 1) FFT + freqs
    w     = np.hanning(len(chunk))
    spec  = np.abs(np.fft.rfft(chunk * w))
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)

    # 2) log-spaced band edges
    f_max = sr / 2
    edges = np.logspace(np.log10(f_min), np.log10(f_max), num=width+1)

    # 3) **peak** per-band instead of mean
    bands = []
    for i in range(width):
        lo, hi = edges[i], edges[i+1]
        idx    = np.where((freqs >= lo) & (freqs < hi))[0]
        if idx.size:
            bands.append(spec[idx].max())
        else:
            bands.append(0.0)

    # 4) to dB relative to file-peak `ref`
    mags_db = 20 * np.log10(np.array(bands) / (ref + 1e-9) + 1e-9)

    # 5) clamp & normalize
    mags_db = np.clip(mags_db, floor_db, None)
    norm    = (mags_db - floor_db) / (-floor_db)
    levels  = (norm * (height - 1)).astype(int)

    # 6) render rows top→bottom
    rows = []
    for row in reversed(range(height)):
        line = ''
        for lvl in levels:
            if lvl >= row:
                if   row > height/1.2: color = 160
                elif row > height/1.7: color = 166
                elif row > height/3:   color = 3
                elif row > 1:          color = 46
                else:                  color = 22
                line += f'\033[38;5;{color}m██'
            else:
                line += '  '
        rows.append(line + '\033[0m')
    return rows

def print_metadata(filepath):
    desired_keys = ["title", "artist", "album", "copyright"]
    meta = MutagenFile(filepath, easy=True)
    if not meta:
        print("No metadata found.")
        return
    print("Metadata:")
    for key in desired_keys:
        value = meta.get(key)
        if value:
            print(f"  {key}: {value[0]}")

def play_and_visualize(filepath):
    # 1) load audio
    y, sr = load_audio(filepath, mono=True)
    print("\033[H\033[J", end="")

    # 2) ensure sample rate
    try:
        sd.check_output_settings(samplerate=sr, channels=1)
    except sd.PortAudioError:
        print(f"Warning: Sample rate {sr} not supported. Falling back to 44100 Hz.")
        sr = 44100
        y, _ = load_audio(filepath, sr=sr, mono=True)

    if y.size == 0:
        print("Failed to load audio.")
        return

    # 3) compute global-peak FFT once
    window        = np.hanning(len(y))
    full_spectrum = np.abs(np.fft.rfft(y * window))
    file_peak     = np.percentile(full_spectrum, 90)

    # 4) print metadata & duration
    print_metadata(filepath)
    duration = len(y) / sr
    print(f"Duration: {duration:.2f} seconds\n")

    # 5) prepare audio callback
    blocksize = 1024
    def callback(outdata, frames, time_info, status):
        idx   = callback.idx
        chunk = y[idx:idx+frames]
        if len(chunk) < frames:
            outdata[:len(chunk), 0] = chunk
            outdata[len(chunk):, 0] = 0
            raise sd.CallbackStop()
        outdata[:, 0] = chunk
        callback.idx += frames
    callback.idx = 0

    stream = sd.OutputStream(
        samplerate=sr,
        channels=1,
        callback=callback,
        blocksize=blocksize
    )

    # 6) visual loop synced to wall-clock time
    hop    = sr // 10        # 0.1s chunks
    fps    = 120             # you can set this as you like
    frames = int(np.ceil(duration * fps))
    meta_shown = False
    with stream:
        start_time = time.time()
        for frame in range(frames):
            elapsed = time.time() - start_time
            idx     = int(elapsed * sr)
            if idx >= len(y):
                break

            chunk = y[idx : idx + hop]
            bars  = render_spectrum_vertical(
                chunk,
                width=60, height=18, sr=sr,
                floor_db=-60.0, f_min=20.0,
                ref=file_peak
            )

            # redraw
            print("\033[H" * 9, end="")
            for line in bars:
                print(f"│{line}")
            print(f"└ Elapsed: {elapsed:.2f}s / {duration:.2f}s".ljust(80))
            if not meta_shown:
                print_metadata(filepath)
                meta_shown = True

            # wait for next frame tick
            next_tick = start_time + (frame + 1) / fps
            sleep_sec = next_tick - time.time()
            if sleep_sec > 0:
                time.sleep(sleep_sec)

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
