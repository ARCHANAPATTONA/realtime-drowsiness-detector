"""
generate_alarm.py
-----------------
Utility: generates a synthetic alarm WAV file at assets/alarm.wav
Run once before first use: python generate_alarm.py
"""

import os
import struct
import math

def write_wav(filename, samples, sample_rate=44100, num_channels=1, bits=16):
    num_samples = len(samples)
    byte_rate   = sample_rate * num_channels * bits // 8
    block_align = num_channels * bits // 8
    data_size   = num_samples * block_align

    with open(filename, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I",  16))           # chunk size
        f.write(struct.pack("<H",   1))           # PCM
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        for s in samples:
            f.write(struct.pack("<h", int(max(-32768, min(32767, s)))))

def generate_alarm(path="assets/alarm.wav"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sr       = 44100
    duration = 2.5    # seconds
    volume   = 28000

    samples = []
    for i in range(int(sr * duration)):
        t = i / sr
        # Dual-tone alarm: 880 Hz + 1100 Hz, with 4 Hz amplitude modulation
        tone = (math.sin(2 * math.pi * 880  * t) +
                math.sin(2 * math.pi * 1100 * t)) / 2
        env  = abs(math.sin(2 * math.pi * 4 * t))    # pulse envelope
        samples.append(tone * env * volume)

    write_wav(path, samples)
    print(f"Alarm sound generated: {path}")

if __name__ == "__main__":
    generate_alarm()
