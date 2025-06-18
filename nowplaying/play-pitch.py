import sys
import time
import numpy as np
import sounddevice as sd
from mutagen import File as MutagenFile
from blessed import Terminal

from .utils import load_audio
fast = False
pretty = False
BLOCKS = " ▁▂▃▄▅▆▇█"  # Optional

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

def play_and_visualize(filepath):
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

    color_steps = [160, 166, 3, 46, 22]

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
                        char = "█" if spectrum_height - row <= lvl and fast else "███" if spectrum_height - row <= lvl and pretty else "██" if spectrum_height - row <= lvl else " " if fast else "   " if pretty else "  "
                        if char.strip():
                            line += term.color(color_code)(char)
                        else:
                            line += char
                    output_lines.append(term.move(row, 0) + line)

                output_lines.append(term.move(spectrum_height, 0) +
                                    f"└ Elapsed: {elapsed:.2f}s / {duration:.2f}s".ljust(80))

                for i, (key, value) in enumerate(metadata.items()):
                    output_lines.append(term.move(spectrum_height + 2 + i, 0) + f"{key}: {value}")

                # Combine and flush all at once
                full_frame = term.move(0, 0) + "".join(output_lines)
                sys.stdout.write(full_frame)
                sys.stdout.flush()

                time.sleep(max(0, (start_time + (frame + 1) / fps) - time.time()))

    print(term.move(spectrum_height + 7, 0) + term.green("Playback finished."))

def main():
    if len(sys.argv) < 2:
        print("Usage: python play_pitch.py <audiofile>")
        sys.exit(1)
    try:
        if len(sys.argv) > 2 and sys.argv[2] == "--fast":
            global fast
            fast = True
        elif len(sys.argv) > 2 and sys.argv[2] == "--pretty":
            global pretty
            pretty = True
        play_and_visualize(sys.argv[1])
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()