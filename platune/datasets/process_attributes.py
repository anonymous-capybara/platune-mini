import gin
import numpy as np
import torch
from scipy.interpolate import interp1d


def get_midi_notes(midi):
    # only for monophonic solo midi
    assert len(midi.instruments) == 1
    for instrument in midi.instruments:
        pitches = []
        onsets = []
        offsets = []
        velocities = []
        for note in instrument.notes:
            pitches.append(note.pitch)
            onsets.append(note.start)
            offsets.append(note.end)
            velocities.append(note.velocity)

    return np.array(pitches).astype(np.int32), np.array(onsets), np.array(offsets), np.array(velocities)


def get_melody_onsets(midi, audio_length, sr = 44100):
    pitch, onset, offset, velocity = get_midi_notes(midi)

    melody = np.zeros((audio_length, )).astype(np.int32)
    onsets_signal = np.zeros(audio_length).astype(np.int32)
    dynamics = np.zeros(audio_length).astype(np.int32)

    onset_sample_positions = np.round(onset * sr).astype(int)
    offset_sample_positions = np.round(offset * sr).astype(int)
    onset_sample_positions = np.clip(onset_sample_positions, 0, audio_length - 1)
    offset_sample_positions = np.clip(offset_sample_positions, 0, audio_length)
    for i in range(len(pitch)):
        melody[onset_sample_positions[i]:offset_sample_positions[i]] = pitch[i]
        dynamics[onset_sample_positions[i]:offset_sample_positions[i]] = velocity[i]
    
    onsets_signal[onset_sample_positions] = 1

    return melody, onsets_signal, dynamics


def process_melody(melody, default_midi_note=60):
    last_note = default_midi_note
    new_melody = melody.copy()
    for i in range(melody.shape[-1]):
        current_note = melody[i]
        if current_note > 0:
            last_note = current_note
        new_melody[i] = last_note
    return new_melody


def downsample_to_latent_sample_rate(melody, z_length):
    z_indices = np.linspace(0, 1, z_length)
    current_indices = np.linspace(0, 1, melody.shape[0])
    interp_func = interp1d(current_indices, melody, kind='nearest')
    melody_downsampled = interp_func(z_indices)
    return melody_downsampled.astype(np.int32)


@gin.configurable
def process_midi_attributes(x, instrument_val, z_length, num_signal, ae_ratio, pitch_note_values, octave_boundaries, instruments_values, dynamics_boundaries):
    attr = {}

    melody, onsets_signal, dynamics = get_melody_onsets(midi=x, audio_length=num_signal, sr=44100)

    melody_processed = process_melody(melody)

    split_per_frame = onsets_signal.reshape(len(onsets_signal) // ae_ratio, -1)
    onsets_downsampled = np.any(split_per_frame[:] == 1, axis=1).astype(int)
    attr['onsets'] = onsets_downsampled[:z_length]

    melody_downsampled = downsample_to_latent_sample_rate(melody, z_length)
    attr['melody'] = melody_downsampled

    melody_processed_downsampled = downsample_to_latent_sample_rate(melody_processed, z_length)
    attr['melody_processed'] = melody_processed_downsampled

    attr['pitch'] = (melody_downsampled % len(pitch_note_values)).astype(int)
    attr['octave'] = (torch.bucketize(torch.from_numpy(melody_downsampled), torch.tensor(octave_boundaries)) - 1).numpy()

    attr['pitch_processed'] = (melody_processed_downsampled % len(pitch_note_values)).astype(int)
    attr['octave_processed'] = (torch.bucketize(torch.from_numpy(melody_processed_downsampled), torch.tensor(octave_boundaries)) - 1).numpy()

    attr['velocity'] = downsample_to_latent_sample_rate(dynamics, z_length)
    attr['dynamics'] = (torch.bucketize(torch.from_numpy(attr['velocity']), torch.tensor(dynamics_boundaries))).numpy()

    attr['instrument'] = np.full((z_length,), instruments_values.index(instrument_val))

    return attr


def process_pt_signal(pt_signal, default_pt_class=0):
    """Fill-forward for playing technique: empty (0) frames inherit the last seen PT class."""
    last_pt = default_pt_class
    new_pt = pt_signal.copy()
    for i in range(pt_signal.shape[-1]):
        current_pt = pt_signal[i]
        if current_pt > 0:
            last_pt = current_pt
        new_pt[i] = last_pt
    return new_pt


@gin.configurable
def process_playing_technique(playing_techniques, chunk_number, num_signal, z_length, ae_ratio, sr, pt_class_values, default_pt_class='Port'):
    """Build a sample-level playing-technique signal from per-note PT1-1 labels,
    apply fill-forward, and downsample to latent time resolution.

    Parameters
    ----------
    playing_techniques : list of (onset, duration, pt_label) tuples
        Per-note playing technique data for the *full* audio file.
    chunk_number : int
        Which chunk to extract.
    num_signal : int
        Chunk length in samples.
    z_length : int
        Number of latent frames per chunk.
    ae_ratio : int
        Samples per latent frame.
    sr : int
        Sample rate.
    pt_class_values : list of str
        Ordered list of PT1-1 class names (e.g. ['Port', 'Vibrato', ...]).
    default_pt_class : str
        Default PT class for fill-forward initialisation.

    Returns
    -------
    np.ndarray of int32, shape [z_length]
    """
    tstart = chunk_number * num_signal / sr
    tend = (chunk_number + 1) * num_signal / sr

    default_idx = pt_class_values.index(default_pt_class) if default_pt_class in pt_class_values else 0

    # Build sample-level PT signal for this chunk (0 = unset)
    pt_signal = np.zeros(num_signal, dtype=np.int32)

    for onset, duration, pt_label in playing_techniques:
        note_end = onset + duration
        # Skip notes outside this chunk
        if note_end <= tstart or onset >= tend:
            continue

        # Clip to chunk boundaries and convert to sample positions
        local_start = max(0.0, onset - tstart)
        local_end = min(tend - tstart, note_end - tstart)
        start_sample = int(round(local_start * sr))
        end_sample = int(round(local_end * sr))
        start_sample = max(0, min(start_sample, num_signal))
        end_sample = max(0, min(end_sample, num_signal))

        if pt_label and pt_label in pt_class_values:
            class_idx = pt_class_values.index(pt_label)
        else:
            class_idx = 0  # will be overwritten by fill-forward

        if pt_label and pt_label in pt_class_values:
            pt_signal[start_sample:end_sample] = class_idx + 1  # +1 so 0 remains "unset"

    # Fill-forward (0 = unset → inherit last seen PT)
    pt_filled = process_pt_signal(pt_signal, default_pt_class=default_idx + 1)
    # Shift back: stored classes are 0-based indices into pt_class_values
    pt_filled = pt_filled - 1
    pt_filled = np.clip(pt_filled, 0, len(pt_class_values) - 1)

    # Downsample to latent rate
    pt_downsampled = downsample_to_latent_sample_rate(pt_filled, z_length)
    return pt_downsampled
