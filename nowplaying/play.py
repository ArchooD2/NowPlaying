import sys
import time
import threading
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from mutagen import File as MutagenFile
from .utils import load_audio

def print_metadata(filepath):
    meta = MutagenFile(filepath, easy=True)
    if meta is None:
        print("No metadata found.")
        return
    print("Metadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

def plot_waveform(y, sr, cursor_sample):
    plt.clf()
    window = sr  # 1 second window
    start = max(0, cursor_sample - window // 2)
    end = min(len(y), start + window)
    plt.plot(np.arange(start, end) / sr, y[start:end])
    plt.axvline(cursor_sample / sr, color='r', linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform (red = current position)")
    plt.tight_layout()
    plt.pause(0.01)

def play_and_visualize(filepath):
    y, sr = load_audio(filepath, mono=True)
    if y.size == 0:
        print("Failed to load audio.")
        return

    print_metadata(filepath)
    duration = len(y) / sr
    print(f"Duration: {duration:.2f} seconds\n")

    plt.ion()
    fig = plt.figure(figsize=(8, 3))

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

    with stream:
        while cursor[0] < len(y):
            elapsed = cursor[0] / sr
            print(f"\rElapsed: {elapsed:.2f}s / {duration:.2f}s", end="")
            plot_waveform(y, sr, cursor[0])
            plt.gcf().canvas.flush_events()
            time.sleep(0.05)
    print("\nPlayback finished.")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audio_player_cli.py <audiofile>")
        sys.exit(1)
    play_and_visualize(sys.argv[1])