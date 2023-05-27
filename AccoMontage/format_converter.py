import pretty_midi as pyd
import numpy as np
import sys
sys.path.append('AccoMontage/chord_recognition')
from main import transcribe_cb1000_midi
from scipy.interpolate import interp1d
import mir_eval


def expand_chord(chord, shift=0, relative=False):
    """
    expand 14-D chord feature to 36-D
    For detail, see Z. Wang et al., "Learning interpretable representation for controllable polyphonic music generation," ISMIR 2020.
    """
    # chord = np.copy(chord)
    root = (chord[0] + shift) % 12
    chroma = np.roll(chord[1: 13], shift)
    bass = (chord[13] + shift) % 12
    root_onehot = np.zeros(12)
    root_onehot[int(root)] = 1
    bass_onehot = np.zeros(12)
    bass_onehot[int(bass)] = 1
    return np.concatenate([root_onehot, chroma, bass_onehot])


def midi2matrix(track, quaver):
    """
    quantize a PrettyMIDI track based on specified quavers.
    The quantized result is a (T, 128) format defined in defined in Z. Wang et al., "Learning interpretable representation for controllable polyphonic music generation," ISMIR 2020.
    """
    #program = track.program
    pr_matrix = np.zeros((len(quaver), 128))
    for note in track.notes:
        note_start = np.argmin(np.abs(quaver - note.start))
        note_end =  np.argmin(np.abs(quaver - note.end))
        if note_end == note_start:
            note_end = min(note_start + 1, len(quaver) - 1)
        pr_matrix[note_start, note.pitch] = note_end - note_start
    return pr_matrix

def ec2vae_mel_format(pr_matrix):
    """
    convert (T, 128) melody format to (T, 130) format.
    (T, 128) format defined in Z. Wang et al., "Learning interpretable representation for controllable polyphonic music generation," ISMIR 2020.
    (T, 130) format defined in R. Yang et al., "Deep music analogy via latent representation disentanglement," ISMIR 2019.
    """
    hold_pitch = 128
    rest_pitch = 129
    melody_roll = np.zeros((len(pr_matrix), 130))
    for t, p in zip(*np.nonzero(pr_matrix)):
        dur = int(pr_matrix[t, p])
        melody_roll[t, p] = 1
        melody_roll[t+1:t+dur, hold_pitch] = 1
    melody_roll[np.nonzero(1 - np.sum(melody_roll, axis=1))[0], rest_pitch] = 1
    return melody_roll


def leadsheet2matrix(path, melody_track_ID=0):
    """
    Tokenize and quantize a lead sheet (a melody track with a chord track).
    The input can also be an arbiturary MIDI file with multiple accompaniment tracks. 
    The first track is by default taken as melody. Otherwise, specify melody_track_ID (counting from zero)
    """
    ACC = 4 #quantize at 1/16 beat
    midi = pyd.PrettyMIDI(path)
    beats = midi.get_beats()
    beats = np.append(beats, beats[-1] + (beats[-1] - beats[-2]))
    quantize = interp1d(np.array(range(0, len(beats))) * ACC, beats, kind='linear')
    quaver = quantize(np.array(range(0, (len(beats) - 1) * ACC)))
    melody_roll = ec2vae_mel_format(midi2matrix(midi.instruments[melody_track_ID], quaver))

    chord_detection = transcribe_cb1000_midi(path)
    chord_roll = np.zeros((len(melody_roll), 14))
    for chord in chord_detection:
        chord_start, chord_end, chord_symbol = chord
        chord_start = np.argmin(np.abs(quaver - chord_start))
        chord_end =  np.argmin(np.abs(quaver - chord_end))
        chord_root, bit_map, bass = mir_eval.chord.encode(chord_symbol)
        chord = np.concatenate([np.array([chord_root]), np.roll(bit_map, shift=int(chord_root)), np.array([bass])])
        chord_roll[chord_start: chord_end] = chord
    chord_roll[np.sum(chord_roll, axis=1)==0, 0]=-1
    chord_roll[np.sum(chord_roll, axis=1)==0, -1]=-1

    return melody_roll, chord_roll


def melody_matrix2data(melody_matrix, tempo=120, start_time=0.0):
    """reconstruct melody from matrix to MIDI"""
    ROLL_SIZE =130
    HOLD_PITCH = 128
    REST_PITCH = 129
    melodyMatrix = melody_matrix[:, :ROLL_SIZE]
    melodySequence = [np.argmax(melodyMatrix[i]) for i in range(melodyMatrix.shape[0])]

    melody_notes = []
    minStep = 60 / tempo / 4
    onset_or_rest = [i for i in range(len(melodySequence)) if not melodySequence[i]==HOLD_PITCH]
    onset_or_rest.append(len(melodySequence))
    for idx, onset in enumerate(onset_or_rest[:-1]):
        if melodySequence[onset] == REST_PITCH:
            continue
        else:
            pitch = melodySequence[onset]
            start = onset * minStep
            end = onset_or_rest[idx+1] * minStep
            noteRecon = pyd.Note(velocity=100, pitch=pitch, start=start_time+start, end=start_time+end)
            melody_notes.append(noteRecon)
    melody = pyd.Instrument(program=pyd.instrument_name_to_program('Acoustic Grand Piano'))
    melody.notes = melody_notes
    return melody


def chord_matrix2data(chordMatrix, tempo=120, start_time=0.0, get_list=False):
    """reconstruct chord from matrix to MIDI"""
    chordSequence = []
    for i in range(chordMatrix.shape[0]):
        chordSequence.append(''.join([str(int(j)) for j in chordMatrix[i]]))
    minStep = 60 / tempo / 4    #16th quantization
    chord_notes = []
    onset_or_rest = [0]
    onset_or_rest_ = [i for i in range(1, len(chordSequence)) if chordSequence[i] != chordSequence[i-1] ]
    onset_or_rest = onset_or_rest + onset_or_rest_
    onset_or_rest.append(len(chordSequence))
    for idx, onset in enumerate(onset_or_rest[:-1]):
        chordset = [int(i) for i in chordSequence[onset]]
        start = onset * minStep
        end = onset_or_rest[idx+1] * minStep
        for note, value in enumerate(chordset):
            if value == 1:
                noteRecon = pyd.Note(velocity=100, pitch=note+4*12, start=start_time+start, end=start_time+end)
                chord_notes.append(noteRecon)
    chord = pyd.Instrument(program=pyd.instrument_name_to_program('Acoustic Grand Piano'))
    chord.notes = chord_notes
    return chord
    

def matrix2leadsheet(leadsheet, tempo=120, start_time=0.0):
    """reconstruct lead sheet from matrix to MIDI"""
    #leadsheet: (T, 142)
    midi = pyd.PrettyMIDI(initial_tempo=tempo)
    midi.instruments.append(melody_matrix2data(leadsheet[:, :130], tempo, start_time))
    midi.instruments.append(chord_matrix2data(leadsheet[:, 130:], tempo, start_time))
    return midi


def accompany_data2matrix(accompany_track, downbeats):
    """
    quantize a PrettyMIDI track into a (T, 128) format as defined in Wang et al., "Learning interpretable representation for controllable polyphonic music generation," ISMIR 2020.
    This function has the same purpose as midi2matrix().
    """
    time_stamp_sixteenth_reso = []
    delta_set = []
    downbeats = list(downbeats)
    downbeats.append(downbeats[-1] + (downbeats[-1] - downbeats[-2]))
    for i in range(len(downbeats)-1):
        s_curr = round(downbeats[i] * 16) / 16
        s_next = round(downbeats[i+1] * 16) / 16
        delta = (s_next - s_curr) / 16
        for i in range(16):
            time_stamp_sixteenth_reso.append(s_curr + delta * i)
            delta_set.append(delta)
    time_stamp_sixteenth_reso = np.array(time_stamp_sixteenth_reso)

    pr_matrix = np.zeros((time_stamp_sixteenth_reso.shape[0], 128))
    for note in accompany_track.notes:
        onset = note.start
        t = np.argmin(np.abs(time_stamp_sixteenth_reso - onset))
        p = note.pitch
        duration = int(round((note.end - onset) / delta_set[t]))
        pr_matrix[t, p] = duration
    return pr_matrix

def accompany_matrix2data(pr_matrix, tempo=120, start_time=0.0, get_list=False):
    """reconstruct a (T, 128) polyphony from magtrix to MIDI."""
    alpha = 0.25 * 60 / tempo
    notes = []
    for t in range(pr_matrix.shape[0]):
        for p in range(128):
            if pr_matrix[t, p] >= 1:
                s = alpha * t + start_time
                e = alpha * (t + pr_matrix[t, p]) + start_time
                notes.append(pyd.Note(100, int(p), s, e))
    if get_list:
        return notes
    else:
        acc = pyd.Instrument(program=pyd.instrument_name_to_program('Acoustic Grand Piano'))
        acc.notes = notes
        return acc


def grid2pr(grid, max_note_count=16, min_pitch=0, pitch_eos_ind=129):
    """
    convert a (T, max_simu_note, 6) format grid into (T, 128 polyphony).
    The (T, max_simu_note, 6) format is defined in Wang et al., "PIANOTREE VAE: Structured Representation Learning for Polyphonic Music," ISMIR 2020.
    The (T, 128 polyphony) format is defined in Wang et al., "Learning interpretable representation for controllable polyphonic music generation," ISMIR 2020.
    """
    #grid: (time, max_simu_note, 6)
    if grid.shape[1] == max_note_count:
        grid = grid[:, 1:]
    pr = np.zeros((grid.shape[0], 128), dtype=int)
    for t in range(grid.shape[0]):
        for n in range(grid.shape[1]):
            note = grid[t, n]
            if note[0] == pitch_eos_ind:
                break
            pitch = note[0] + min_pitch
            dur = int(''.join([str(_) for _ in note[1:]]), 2) + 1
            pr[t, pitch] = dur
    return pr


def matrix2midi_with_dynamics(pr_matrices, programs, init_tempo=120, time_start=0, ACC=16):
    """
    Reconstruct a multi-track midi from a 3D matrix of shape (Track. Time, 128, 3).
    The last dimension each encoders MIDI pitch, velocity, and control message.
    """
    tracks = []
    for program in programs:
        track_recon = pyd.Instrument(program=int(program), is_drum=False, name=pyd.program_to_instrument_name(int(program)))
        tracks.append(track_recon)

    indices_track, indices_onset, indices_pitch = np.nonzero(pr_matrices[:, :, :, 0])
    alpha = 1 / (ACC // 4) * 60 / init_tempo #timetep between each quntization bin
    for idx in range(len(indices_track)):
        track_id = indices_track[idx]
        onset = indices_onset[idx]
        pitch = indices_pitch[idx]

        start = onset * alpha
        duration = pr_matrices[track_id, onset, pitch, 0] * alpha
        velocity = pr_matrices[track_id, onset, pitch, 1]

        note_recon = pyd.Note(velocity=int(velocity), pitch=int(pitch), start=time_start + start, end=time_start + start + duration)
        tracks[track_id].notes.append(note_recon)

    for idx in range(len(pr_matrices)):
        cc = []
        control_matrix = pr_matrices[idx, :, :, 2]
        for t, n in zip(*np.nonzero(control_matrix >= 0)):
            start = alpha * t
            cc.append(pyd.ControlChange(int(n), int(control_matrix[t, n]), start))
        tracks[idx].control_changes = cc
    
    midi_recon = pyd.PrettyMIDI(initial_tempo=init_tempo)
    midi_recon.instruments = tracks
    return midi_recon
