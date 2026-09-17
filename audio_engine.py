"""
audio_engine.py
================
Real-time Polyphonic Procedural Sound Synthesis Engine.

Key Features:
- Persistent background non-blocking output stream via sounddevice.OutputStream
  (44100 Hz, stereo 2-channels, float32, blocksize 256; ~5.8 ms block duration at 44.1 kHz; actual device/output latency is hardware and driver dependent).
- Active Voice Manager with thread-safe polyphony and automatic voice retirement.
- Additive Synthesis Piano Voice with exponential Attack-Decay (AD) percussive envelope.
- Plucked String Synthesis Guitar Voice via damped harmonic spectrum and chord offsets.
- Soft-clipping master bus using np.tanh to prevent digital saturation.
- Pure procedural generation: zero external audio files (.wav/.mp3).
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    import sounddevice as sd
except ImportError:
    sd = None

logger = logging.getLogger("AudioEngine")


class Voice(ABC):
    """Abstract Base Class for all procedural synthesizers."""

    def __init__(self, sample_rate: int = 44100, pan: float = 0.0) -> None:
        self.sample_rate = sample_rate
        self.pan = float(np.clip(pan, -1.0, 1.0))
        self.left_gain = float(np.cos((self.pan + 1.0) * np.pi / 4.0))
        self.right_gain = float(np.sin((self.pan + 1.0) * np.pi / 4.0))
        self.current_sample = 0
        self._is_finished = False

    @abstractmethod
    def render(self, num_frames: int) -> np.ndarray:
        """
        Renders the next block of stereo audio samples.

        Args:
            num_frames: Number of audio samples to generate.

        Returns:
            np.ndarray of shape (num_frames, 2), dtype float32.
        """
        raise NotImplementedError

    def is_finished(self) -> bool:
        """Returns True if the voice has completed its envelope and can be purged."""
        return self._is_finished


class PianoVoice(Voice):
    """
    Additive Synthesis Piano Voice.

    Mathematical Model:
    - Fundamental Sine: sin(2 * pi * f * t)
    - Octave Harmonic: 0.5 * sin(2 * pi * 2f * t)
    - Subtle Triangle: 0.15 * (2 * |2 * (ft mod 1) - 1| - 1)
    - Envelope: Percussive Attack-Decay (AD) exponential curve:
        Attack A = 0.005s (linear ramp to peak)
        Decay  D = 0.800s (natural acoustic exponential decay towards silence)
        Total  T = 0.805s (voice marks finished upon reaching decay completion)
    """

    def __init__(
        self,
        freq: float,
        velocity: float = 1.0,
        sample_rate: int = 44100,
        pan: float = 0.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate, pan=pan)
        self.freq = max(20.0, float(freq))
        self.velocity = float(np.clip(velocity, 0.05, 1.0))

        # AD envelope parameters (in seconds)
        self.attack_time = 0.005
        self.decay_time = 0.800

        self.attack_samples = int(self.attack_time * self.sample_rate)
        self.decay_samples = int(self.decay_time * self.sample_rate)
        self.total_samples = self.attack_samples + self.decay_samples

        # Decay rate constant for exponential curve: e^(-alpha * t)
        # Drops to ~0.005 at the end of decay_time
        self.decay_alpha = 5.3 / max(1, self.decay_samples)

    def render(self, num_frames: int) -> np.ndarray:
        if self._is_finished:
            return np.zeros((num_frames, 2), dtype=np.float32)

        start_idx = self.current_sample
        end_idx = start_idx + num_frames
        self.current_sample = end_idx

        indices = np.arange(start_idx, end_idx, dtype=np.float32)
        t = indices / self.sample_rate

        # 1. Additive synthesis harmonics
        # Fundamental sine
        fundamental = np.sin(2.0 * np.pi * self.freq * t)
        # 2nd harmonic (octave) at 0.5 amplitude
        octave = 0.5 * np.sin(4.0 * np.pi * self.freq * t)
        # Triangle harmonic for subtle body resonance
        phase = (self.freq * t) % 1.0
        triangle = 0.15 * (2.0 * np.abs(2.0 * phase - 1.0) - 1.0)

        raw_wave = (fundamental + octave + triangle) * (self.velocity * 0.7)

        # 2. Percussive Attack-Decay (AD) Envelope calculation
        envelope = np.zeros(num_frames, dtype=np.float32)

        # Attack phase mask
        attack_mask = indices < self.attack_samples
        if np.any(attack_mask):
            envelope[attack_mask] = indices[attack_mask] / max(1.0, float(self.attack_samples))

        # Decay phase mask
        decay_mask = (indices >= self.attack_samples) & (indices < self.total_samples)
        if np.any(decay_mask):
            decay_indices = indices[decay_mask] - self.attack_samples
            envelope[decay_mask] = np.exp(-self.decay_alpha * decay_indices)

        # Past total duration
        finished_mask = indices >= self.total_samples
        if np.any(finished_mask):
            envelope[finished_mask] = 0.0

        if end_idx >= self.total_samples:
            self._is_finished = True

        mono_signal = raw_wave * envelope

        # Stereo positioning
        stereo = np.empty((num_frames, 2), dtype=np.float32)
        stereo[:, 0] = mono_signal * self.left_gain
        stereo[:, 1] = mono_signal * self.right_gain
        return stereo


class GuitarVoice(Voice):
    """
    Plucked String Synthesis Guitar Voice.

    Mathematical Model:
    Plucked strings exhibit frequency-dependent damping: higher harmonics decay
    much faster than the fundamental. We synthesize a band-limited sawtooth
    waveform across K harmonics, each modulated by an individual exponential decay:
        s(t) = sum_{k=1}^K [ (-1)^(k-1) / k * sin(2*pi*k*f*t) * exp(-t * (gamma_0 + gamma_1 * k^1.6)) ]
    with a fast 2ms initial pluck attack envelope to eliminate startup click.
    """

    def __init__(
        self,
        freq: float,
        velocity: float = 1.0,
        sample_rate: int = 44100,
        pan: float = 0.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate, pan=pan)
        self.freq = max(40.0, float(freq))
        self.velocity = float(np.clip(velocity, 0.1, 1.0))

        # Acoustic pluck decay parameters
        self.gamma_0 = 1.9    # Base decay rate of fundamental
        self.gamma_1 = 0.45   # Exponential damping multiplier for harmonics
        self.pluck_attack_tau = 0.003  # 3ms pluck transient

        # Number of audible harmonics up to Nyquist limit
        max_harmonics = int(min(22, (self.sample_rate * 0.45) // self.freq))
        self.num_harmonics = max(3, max_harmonics)

        # Precompute harmonic weights and decay multipliers
        self.k_vals = np.arange(1, self.num_harmonics + 1, dtype=np.float32)
        self.harmonic_weights = (((-1.0) ** (self.k_vals - 1)) / self.k_vals).astype(np.float32)
        self.decay_rates = (self.gamma_0 + self.gamma_1 * (self.k_vals ** 1.6)).astype(np.float32)

        # Total duration before note is considered silent (~1.8 seconds)
        self.max_duration = 1.8
        self.total_samples = int(self.max_duration * self.sample_rate)

    def render(self, num_frames: int) -> np.ndarray:
        if self._is_finished:
            return np.zeros((num_frames, 2), dtype=np.float32)

        start_idx = self.current_sample
        end_idx = start_idx + num_frames
        self.current_sample = end_idx

        indices = np.arange(start_idx, end_idx, dtype=np.float32)
        t = indices / self.sample_rate

        if end_idx >= self.total_samples:
            self._is_finished = True

        # Pluck attack transient (smooth non-zero rise)
        pluck_envelope = 1.0 - np.exp(-t / self.pluck_attack_tau)

        # Multi-harmonic summation:
        # t is (num_frames,), k is (num_harmonics,) -> outer broadcast
        # 2 * pi * f * k * t: shape (num_harmonics, num_frames)
        phase_matrix = 2.0 * np.pi * self.freq * np.outer(self.k_vals, t)
        sin_matrix = np.sin(phase_matrix)

        # Decay matrix: exp(-decay_rate * t) -> shape (num_harmonics, num_frames)
        decay_matrix = np.exp(-np.outer(self.decay_rates, t))

        # Weight each harmonic
        weighted_harmonics = (
            self.harmonic_weights[:, np.newaxis]
            * sin_matrix
            * decay_matrix
        )
        mono_signal = np.sum(weighted_harmonics, axis=0) * pluck_envelope * (self.velocity * 0.65)

        stereo = np.empty((num_frames, 2), dtype=np.float32)
        stereo[:, 0] = mono_signal * self.left_gain
        stereo[:, 1] = mono_signal * self.right_gain
        return stereo


class CyberSFXVoice(Voice):
    """Procedural futuristic sound effect for spatial gesture actions."""

    def __init__(self, sfx_type: str = "lock", sample_rate: int = 44100, pan: float = 0.0) -> None:
        super().__init__(sample_rate=sample_rate, pan=pan)
        self.sfx_type = sfx_type
        if sfx_type == "lock":
            self.duration = 0.28
        elif sfx_type == "tick":
            self.duration = 0.04
        elif sfx_type == "lightning":
            self.duration = 0.10
        else:
            self.duration = 0.20
        self.total_samples = int(self.duration * self.sample_rate)

    def render(self, num_frames: int) -> np.ndarray:
        if self._is_finished:
            return np.zeros((num_frames, 2), dtype=np.float32)

        start_idx = self.current_sample
        end_idx = start_idx + num_frames
        self.current_sample = end_idx

        if start_idx >= self.total_samples:
            self._is_finished = True
            return np.zeros((num_frames, 2), dtype=np.float32)

        actual_end = min(end_idx, self.total_samples)
        indices = np.arange(start_idx, actual_end, dtype=np.float32)
        t = indices / self.sample_rate

        if self.sfx_type == "lock":
            # Rising two-tone chime (587.33Hz D5 -> 880Hz A5)
            f = np.where(t < 0.12, 587.33, 880.00)
            env = np.where(t < 0.12, np.exp(-18.0 * t), np.exp(-12.0 * (t - 0.12)))
            sig = 0.35 * np.sin(2.0 * np.pi * f * t) * env
        elif self.sfx_type == "tick":
            sig = 0.14 * np.sin(2.0 * np.pi * 1760.0 * t) * np.exp(-80.0 * t)
        elif self.sfx_type == "lightning":
            noise = (np.random.rand(len(t)).astype(np.float32) * 2.0 - 1.0)
            buzz = np.sin(2.0 * np.pi * 140.0 * t)
            sig = 0.22 * (0.6 * noise + 0.4 * buzz) * np.exp(-25.0 * t)
        else:
            sig = 0.20 * np.sin(2.0 * np.pi * 440.0 * t) * np.exp(-10.0 * t)

        if end_idx >= self.total_samples:
            self._is_finished = True

        out = np.zeros((num_frames, 2), dtype=np.float32)
        actual_len = len(sig)
        out[:actual_len, 0] = sig * self.left_gain
        out[:actual_len, 1] = sig * self.right_gain
        return out


class AudioEngine:
    """
    Active Voice Manager and Ultra-Low Latency Procedural Sound Engine.

    Coordinates:
    - Non-blocking sounddevice.OutputStream running at 44.1 kHz, blocksize=256 (~5.8 ms block duration; total output latency is hardware and driver dependent).
    - Concurrent polyphony mixing with soft-limiting master bus.
    - Piano and Guitar chord/string mappings.
    - Automatic cleanup of exhausted voices.
    """

    # Standard 6-String Acoustic Guitar Open Frequencies (E2 to E4)
    GUITAR_OPEN_FREQS: Tuple[float, ...] = (
        82.41,   # String 0: E2 (6th string, thickest)
        110.00,  # String 1: A2 (5th string)
        146.83,  # String 2: D3 (4th string)
        196.00,  # String 3: G3 (3rd string)
        246.94,  # String 4: B3 (2nd string)
        329.63,  # String 5: E4 (1st string, thinnest)
    )

    # Standard Guitar Chord Fret Offsets (semitones above open string).
    # Form: (str0, str1, str2, str3, str4, str5) from 6th (low E) to 1st (high E).
    # None indicates a MUTED string (must not play).
    CHORD_FRETS: Dict[str, Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]] = {
        "C":  (None, 3, 2, 0, 1, 0),     # x 3 2 0 1 0 (Root C on 5th string)
        "G":  (3, 2, 0, 0, 0, 3),        # 3 2 0 0 0 3 (Root G on 6th string)
        "D":  (None, None, 0, 2, 3, 2),  # x x 0 2 3 2 (Root D on 4th string)
        "A":  (None, 0, 2, 2, 2, 0),     # x 0 2 2 2 0 (Root A on 5th string)
        "E":  (0, 2, 2, 1, 0, 0),        # 0 2 2 1 0 0 (Root E on 6th string)
        "Am": (None, 0, 2, 2, 1, 0),     # x 0 2 2 1 0 (Root A on 5th string)
        "Em": (0, 2, 2, 0, 0, 0),        # 0 2 2 0 0 0 (Root E on 6th string)
        "Dm": (None, None, 0, 2, 3, 1),  # x x 0 2 3 1 (Root D on 4th string)
        "F":  (1, 3, 3, 2, 1, 1),        # 1 3 3 2 1 1 (Full barre F major)
    }

    @classmethod
    def get_supported_chords(cls) -> List[str]:
        """Returns the list of all supported guitar chord names."""
        return list(cls.CHORD_FRETS.keys())

    @classmethod
    def is_string_muted(cls, chord_name: str, string_idx: int) -> bool:
        """Returns True if the specified string is muted (x) in the given chord."""
        frets = cls.CHORD_FRETS.get(chord_name, cls.CHORD_FRETS["C"])
        if 0 <= string_idx < len(frets):
            return frets[string_idx] is None
        return True

    @classmethod
    def get_chord_voicing(cls, chord_name: str) -> Tuple[Optional[int], ...]:
        """Returns the 6-string fret offsets for the chord (None for muted)."""
        return cls.CHORD_FRETS.get(chord_name, cls.CHORD_FRETS["C"])

    def __init__(self, sample_rate: int = 44100, block_size: int = 256) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size

        self._active_voices: List[Voice] = []
        self._lock = threading.Lock()

        self._stream: Optional[sd.OutputStream] = None
        self.is_running = False
        self.audio_available = False
        self.current_peak: float = 0.0

        # Hardware audio stream telemetry
        self.underflow_count: int = 0
        self.overflow_count: int = 0
        self.last_callback_status: str = ""

    def start(self) -> bool:
        """
        Initializes and starts the audio output stream.
        Returns True if successful, False if audio device is unavailable.
        """
        if self.is_running:
            return True

        if sd is None:
            logger.warning("sounddevice module not found; running in silent mode.")
            self.is_running = True
            return False

        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                dtype="float32",
                blocksize=self.block_size,
                callback=self._audio_callback,
            )
            self._stream.start()
            self.is_running = True
            self.audio_available = True
            logger.info("AudioEngine started successfully: 44100Hz, blocksize=%d", self.block_size)
            return True
        except Exception as e:
            logger.warning("Failed to open sounddevice output stream: %s. Continuing in silent mode.", e)
            self.is_running = True
            self.audio_available = False
            return False

    def stop(self) -> None:
        """Stops the audio stream and purges voices."""
        self.is_running = False
        with self._lock:
            self._active_voices.clear()

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug("Error closing audio stream: %s", e)
            finally:
                self._stream = None

    def play_piano(self, freq: float, velocity: float = 1.0, pan: float = 0.0) -> None:
        """Triggers an additive synthesis piano note."""
        voice = PianoVoice(
            freq=freq,
            velocity=velocity,
            sample_rate=self.sample_rate,
            pan=pan,
        )
        with self._lock:
            # Prevent excessive voice buildup (cap at 32 voices)
            if len(self._active_voices) >= 32:
                self._active_voices.pop(0)
            self._active_voices.append(voice)

    def play_guitar(self, string_idx: int, chord_name: str = "C", velocity: float = 1.0) -> bool:
        """
        Triggers a plucked string note for a specific guitar string and chord.

        Args:
            string_idx: String index 0 to 5 (0 = 6th string E2, 5 = 1st string E4).
            chord_name: One of 'C', 'G', 'D', 'A', 'E', 'Am', 'Em', 'Dm', 'F'.
            velocity: Pluck velocity (0.1 to 1.0).

        Returns:
            True if sound was played, False if string is muted in this chord.
        """
        string_idx = int(np.clip(string_idx, 0, 5))
        frets = self.CHORD_FRETS.get(chord_name, self.CHORD_FRETS["C"])
        fret_offset = frets[string_idx]

        # Muted string: MUST NOT play or produce sound
        if fret_offset is None:
            return False

        base_freq = self.GUITAR_OPEN_FREQS[string_idx]
        # Equal temperament frequency shift: f = f_0 * 2^(fret / 12)
        chord_freq = base_freq * (2.0 ** (fret_offset / 12.0))

        # Spatial pan from left (-0.6) to right (+0.6) across the 6 strings
        pan = -0.6 + (string_idx / 5.0) * 1.2

        voice = GuitarVoice(
            freq=chord_freq,
            velocity=velocity,
            sample_rate=self.sample_rate,
            pan=pan,
        )
        with self._lock:
            if len(self._active_voices) >= 32:
                self._active_voices.pop(0)
            self._active_voices.append(voice)
        return True

    def play_shape_lock(self) -> None:
        """Triggers futuristic cyber confirmation chime when a sculpted shape locks."""
        voice = CyberSFXVoice(sfx_type="lock", sample_rate=self.sample_rate)
        with self._lock:
            if len(self._active_voices) >= 32:
                self._active_voices.pop(0)
            self._active_voices.append(voice)

    def play_sculpt_tick(self) -> None:
        """Triggers subtle blip during shape expansion."""
        voice = CyberSFXVoice(sfx_type="tick", sample_rate=self.sample_rate)
        with self._lock:
            if len(self._active_voices) >= 32:
                self._active_voices.pop(0)
            self._active_voices.append(voice)

    def play_lightning_buzz(self) -> None:
        """Triggers electric spark crackle when pieces are magnetically attracted."""
        voice = CyberSFXVoice(sfx_type="lightning", sample_rate=self.sample_rate)
        with self._lock:
            if len(self._active_voices) >= 32:
                self._active_voices.pop(0)
            self._active_voices.append(voice)

    def play_fusion_burst(self) -> None:
        """Triggers epic full guitar power chord strum + chime when pieces fuse."""
        for str_idx in range(6):
            self.play_guitar(string_idx=str_idx, chord_name="G", velocity=0.88)
        self.play_shape_lock()

    def get_active_voice_count(self) -> int:
        """Thread-safe query of currently playing synthesizer voices."""
        with self._lock:
            return len(self._active_voices)

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        """
        Real-time PortAudio high-priority callback.
        Accumulates active voices outside the lock to minimize contention, applies soft clipping,
        and records hardware buffer status telemetry.
        """
        if status:
            if status.output_underflow:
                self.underflow_count += 1
            if status.output_overflow:
                self.overflow_count += 1
            self.last_callback_status = str(status)

        # Snapshot active voices under a short critical section lock
        with self._lock:
            voices_to_render = self._active_voices[:]

        mix_buffer = np.zeros((frames, 2), dtype=np.float32)
        finished_voices: List[Voice] = []

        # Synthesis DSP computation executes outside lock
        for voice in voices_to_render:
            rendered = voice.render(frames)
            mix_buffer += rendered
            if voice.is_finished():
                finished_voices.append(voice)

        # Purge finished voices briefly under lock
        if finished_voices:
            with self._lock:
                finished_set = set(finished_voices)
                self._active_voices = [v for v in self._active_voices if v not in finished_set]

        # Numerical safety: sanitize any non-finite values (NaN / Inf)
        if not np.all(np.isfinite(mix_buffer)):
            np.nan_to_num(mix_buffer, copy=False)

        # Master soft-limiting via hyperbolic tangent to prevent clipping
        np.tanh(mix_buffer, out=outdata)
        peak = float(np.max(np.abs(outdata)))
        self.current_peak = 0.82 * self.current_peak + 0.18 * peak
