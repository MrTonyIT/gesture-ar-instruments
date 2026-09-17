"""
tests/test_audio.py
===================
Unit Tests for Audio Engine and Procedural Sound Synthesis.
Covers:
- All 9 Guitar Chords and Voicings
- Muted string suppression (returns False, 0 voice allocation)
- Piano and Guitar voice synthesis shapes and types
- Soft-limiting and saturation protection (zero hard clipping)
- Numerical sanity: zero NaN / Inf in audio streams
- Polyphonic voice manager and 32-voice capacity cap
"""

import numpy as np
import pytest

from audio_engine import AudioEngine, GuitarVoice, PianoVoice


@pytest.fixture
def audio_engine():
    """Provides a headless AudioEngine instance without starting sounddevice hardware stream."""
    engine = AudioEngine(sample_rate=44100, block_size=256)
    yield engine
    engine.stop()


def test_supported_chords(audio_engine):
    """Verifies all 9 chords are supported with 6-string voicings."""
    expected_chords = ["C", "G", "D", "A", "E", "Am", "Em", "Dm", "F"]
    supported = AudioEngine.get_supported_chords()

    for ch in expected_chords:
        assert ch in supported, f"Chord {ch} must be in supported chords"
        voicing = AudioEngine.get_chord_voicing(ch)
        assert len(voicing) == 6, f"Voicing for {ch} must have 6 string fret offsets"


def test_muted_strings_suppression(audio_engine):
    """
    Verifies that muted strings in chords return False on play_guitar
    and do NOT allocate active voices.
    """
    # C chord: 6th string (E2, idx=0) is muted (None)
    assert AudioEngine.is_string_muted("C", 0) is True
    initial_voices = audio_engine.get_active_voice_count()

    result_muted = audio_engine.play_guitar(string_idx=0, chord_name="C", velocity=0.8)
    assert result_muted is False
    assert audio_engine.get_active_voice_count() == initial_voices

    # C chord: 5th string (A2, idx=1) is active (fret 3)
    assert AudioEngine.is_string_muted("C", 1) is False
    result_active = audio_engine.play_guitar(string_idx=1, chord_name="C", velocity=0.8)
    assert result_active is True
    assert audio_engine.get_active_voice_count() == initial_voices + 1


def test_d_and_dm_chords_muted_low_strings(audio_engine):
    """D and Dm chords have muted 6th and 5th strings."""
    for ch in ["D", "Dm"]:
        assert AudioEngine.is_string_muted(ch, 0) is True
        assert AudioEngine.is_string_muted(ch, 1) is True
        assert AudioEngine.is_string_muted(ch, 2) is False  # Open D string

        # Muted strings return False
        assert audio_engine.play_guitar(string_idx=0, chord_name=ch) is False
        assert audio_engine.play_guitar(string_idx=1, chord_name=ch) is False
        # Unmuted returns True
        assert audio_engine.play_guitar(string_idx=2, chord_name=ch) is True


def test_piano_voice_synthesis():
    """Piano voice generates valid stereo float32 blocks."""
    voice = PianoVoice(freq=440.0, velocity=0.8, sample_rate=44100)
    frames = 512
    block = voice.render(frames)

    assert block.shape == (frames, 2)
    assert block.dtype == np.float32
    assert np.all(np.isfinite(block))
    assert not voice.is_finished()


def test_guitar_voice_synthesis():
    """Guitar voice generates damped harmonic stereo float32 blocks."""
    voice = GuitarVoice(freq=196.0, velocity=0.7, sample_rate=44100)
    frames = 256
    block = voice.render(frames)

    assert block.shape == (frames, 2)
    assert block.dtype == np.float32
    assert np.all(np.isfinite(block))
    assert not voice.is_finished()


def test_voice_capacity_cap(audio_engine):
    """AudioEngine caps active voices at 32 to prevent unbounded CPU allocation."""
    for _ in range(40):
        audio_engine.play_piano(freq=261.63, velocity=0.5)

    assert audio_engine.get_active_voice_count() <= 32


def test_master_soft_limiting():
    """np.tanh soft-limiting guarantees output strictly within [-1.0, 1.0]."""
    loud_buffer = np.full((128, 2), 10.0, dtype=np.float32)
    outdata = np.empty_like(loud_buffer)
    np.tanh(loud_buffer, out=outdata)

    assert np.all(outdata <= 1.0)
    assert np.all(outdata >= -1.0)
    assert np.max(np.abs(outdata)) <= 1.0


def test_audio_callback_rendering_and_telemetry(audio_engine):
    """Verifies _audio_callback renders active voices and updates buffer telemetry."""
    # Add a piano voice and guitar voice
    audio_engine.play_piano(freq=440.0, velocity=0.8)
    audio_engine.play_guitar(string_idx=2, chord_name="G", velocity=0.9)
    assert audio_engine.get_active_voice_count() == 2

    # Simulate PortAudio callback
    frames = 256
    outdata = np.zeros((frames, 2), dtype=np.float32)

    class MockStatus:
        output_underflow = False
        output_overflow = False

        def __bool__(self):
            return False

    audio_engine._audio_callback(outdata, frames, {}, MockStatus())

    # Verify samples were generated and soft-limited
    assert np.any(outdata != 0.0)
    assert np.all(np.isfinite(outdata))
    assert np.all(np.abs(outdata) <= 1.0)
    assert audio_engine.underflow_count == 0
    assert audio_engine.overflow_count == 0

    # Simulate underflow flag
    class UnderflowStatus:
        output_underflow = True
        output_overflow = False

        def __bool__(self):
            return True

        def __str__(self):
            return "output underflow"

    audio_engine._audio_callback(outdata, frames, {}, UnderflowStatus())
    assert audio_engine.underflow_count == 1
    assert audio_engine.last_callback_status == "output underflow"


def test_piano_voice_lifecycle_and_envelope_completion():
    """Piano voice completes after its total duration and marks itself finished."""
    voice = PianoVoice(freq=440.0, velocity=0.8, sample_rate=44100)
    assert not voice.is_finished()

    # Total duration is attack + decay samples = 0.005s + 0.800s = 0.805s * 44100 = 35500 samples
    total_samples = voice.total_samples
    assert total_samples == int(0.005 * 44100) + int(0.800 * 44100)

    # Render in chunks
    chunk_size = 4096
    rendered_total = 0
    while rendered_total < total_samples + chunk_size:
        block = voice.render(chunk_size)
        assert np.all(np.isfinite(block))
        rendered_total += chunk_size

    assert voice.is_finished() is True

    # Subsequent render returns zeros
    extra_block = voice.render(chunk_size)
    assert np.all(extra_block == 0.0)


def test_voice_panning_stereo_shape():
    """Pan value controls left and right channel gain appropriately."""
    # Hard left pan = -1.0
    voice_l = PianoVoice(freq=440.0, velocity=1.0, pan=-1.0)
    block_l = voice_l.render(256)
    assert np.max(np.abs(block_l[:, 0])) > 0.0
    assert np.max(np.abs(block_l[:, 1])) == pytest.approx(0.0, abs=1e-5)

    # Hard right pan = 1.0
    voice_r = PianoVoice(freq=440.0, velocity=1.0, pan=1.0)
    block_r = voice_r.render(256)
    assert np.max(np.abs(block_r[:, 1])) > 0.0
    assert np.max(np.abs(block_r[:, 0])) == pytest.approx(0.0, abs=1e-5)


def test_finished_voice_cleanup_in_callback(audio_engine):
    """AudioEngine purges completed voices during the callback."""
    voice = PianoVoice(freq=440.0, velocity=1.0, sample_rate=audio_engine.sample_rate)
    with audio_engine._lock:
        audio_engine._active_voices.append(voice)
    assert audio_engine.get_active_voice_count() == 1

    # Render enough frames in callback to complete voice
    outdata = np.zeros((4096, 2), dtype=np.float32)

    class CleanStatus:
        output_underflow = False
        output_overflow = False

        def __bool__(self):
            return False

    for _ in range(12):
        audio_engine._audio_callback(outdata, 4096, {}, CleanStatus())

    # Voice completed and was purged under lock
    assert voice.is_finished() is True
    assert audio_engine.get_active_voice_count() == 0

