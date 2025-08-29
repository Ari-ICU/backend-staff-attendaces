import numpy as np
import wave
import os
import struct

# Create a richer chime sound
def create_chime(filename, freq=600, duration=0.6, decay=3.0):
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Base sine wave
    base = np.sin(2 * np.pi * freq * t)

    # Add first harmonic (octave) and a small second harmonic
    harmonic1 = 0.5 * np.sin(2 * np.pi * freq * 2 * t)
    harmonic2 = 0.25 * np.sin(2 * np.pi * freq * 3 * t)

    # Slight frequency glide (optional)
    glide = np.sin(2 * np.pi * (freq + 20 * t) * t)

    # Combine waves and apply exponential decay
    data = (base + harmonic1 + harmonic2 + 0.2 * glide) * np.exp(-t * decay)
    data = (data / np.max(np.abs(data)) * 32767).astype(np.int16)

    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(struct.pack('<' + 'h' * len(data), *data))

# Generate sounds
os.makedirs('sounds', exist_ok=True)
create_chime('sounds/check-in.wav', freq=600, decay=2.5)
create_chime('sounds/check-out.wav', freq=400, decay=3.0)
create_chime('sounds/leave.wav', freq=500, decay=3.5)
