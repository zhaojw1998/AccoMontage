import os
import numpy as np
import pretty_midi as pyd
from dtw import *
from scipy import interpolate
from tqdm import tqdm
import mir_eval
from scipy import stats
import pandas as pd

def split_phrases(segmentation):
    phrases = []
    lengths = []
    current = 0
    while segmentation[current] != '\n':
        if segmentation[current].isalpha():
            j = 1
            while not (segmentation[current + j].isalpha() or segmentation[current + j] == '\n'):
                j += 1
            phrases.append(segmentation[current])
            lengths.append(int(segmentation[current+1: current+j]))
            current += j
    return [(phrases[i], lengths[i], sum(lengths[:i])) for i in range(len(phrases))]  


def matrix2midi_with_dynamics(pr_matrices, programs, init_tempo=120, time_start=0, ACC=16):
    """
    Reconstruct a multi-track midi from a 3D matrix of shape (Track. Time, 128, 3).
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

def get_ec2_melody(pr_matrix):
    hold_pitch = 128
    rest_pitch = 129

    piano_roll = np.zeros((len(pr_matrix), 130))
    for t, p in zip(*np.nonzero(pr_matrix)):
        dur = int(pr_matrix[t, p])
        piano_roll[t, p] = 1
        piano_roll[t+1:t+dur, hold_pitch] = 1
    piano_roll[np.nonzero(1 - np.sum(piano_roll, axis=1))[0], rest_pitch] = 1
    return piano_roll



SEGMENTATION_ROOT = 'hierarchical-structure-analysis/POP909'    # https://github.com/Dsqvival/hierarchical-structure-analysis
POP909_MIDI_ROOT = '../POP909 Dataset (MIDI)'                   # https://github.com/music-x-lab/POP909-Dataset

df = pd.read_excel(f"{POP909_MIDI_ROOT}/index.xlsx")
phrase_melody = []
phrase_acc = []
phrase_chord = []
phrase_velocity = []
phrase_cc = []
for song in os.listdir(SEGMENTATION_ROOT):
    meta_data = df[df.song_id == int(song)]
    num_beats = meta_data.num_beats_per_measure.values[0]
    num_quavers = meta_data.num_quavers_per_beat.values[0]
    if int(num_beats) == 3 or int(num_quavers) == 3:
        continue
    try:
        melody = np.loadtxt(os.path.join(SEGMENTATION_ROOT, song, 'melody.txt'))
    except OSError:
        continue
    melody[:, 1] = np.cumsum(melody[:, 1])
    melody[1:, 1] = melody[:-1, 1]
    melody = melody[1:]
    #print(melody[:, 1])
    melody_notes = []
    for note in melody:
        if note[0] > 0:
            melody_notes.append(note)
    
    midi = pyd.PrettyMIDI(os.path.join(POP909_MIDI_ROOT, song, f'{song}.mid'))
    time_record = []
    midi_notes = []
    for note in midi.instruments[0].notes:
        if not note.start in time_record:
            midi_notes.append(note)
            time_record.append(note.start)

    alignment = dtw([int(note[0]) for note in melody_notes], [note.pitch for note in midi_notes], keep_internals=True)

    melody_note_indices = alignment.index1
    midi_note_indices = alignment.index2
    quaver = []
    time = []
    for idx in range(1, len(melody_note_indices)-1):
        if (melody_note_indices[idx] == melody_note_indices[idx-1]) \
            or (melody_note_indices[idx] == melody_note_indices[idx+1]) \
            or (midi_note_indices[idx] == midi_note_indices[idx-1]) \
            or (midi_note_indices[idx] == midi_note_indices[idx+1]):
            continue
        quaver.append(melody_notes[melody_note_indices[idx]][1])
        time.append(midi_notes[midi_note_indices[idx]].start)
    
    f = interpolate.interp1d(time, quaver, bounds_error=False, fill_value='extrapolate')

    #import matplotlib.pyplot as plt
    #plt.plot(time, quaver, 'o', time, quaver, '-')
    #plt.show()

    with open(os.path.join(SEGMENTATION_ROOT, song, 'human_label1.txt'), 'r') as file:
        segmentation = file.readlines()[0]
    print(song, segmentation)
    if not '\n' in segmentation:
        segmentation += '\n'
    segmentation = split_phrases(segmentation)
    
    tracks = np.concatenate([np.zeros((3, (segmentation[-1][-1] + segmentation[-1][-2]) * 16, 128, 2)), \
                                -1 * np.ones((3, (segmentation[-1][-1] + segmentation[-1][-2]) * 16, 128, 2))], \
                            axis=-1 \
                            )
    for idx, track in enumerate(midi.instruments):
        for note in track.notes:
            start = int(np.round(f(note.start)))
            if start >= tracks.shape[1]:
                break
            end = int(np.round(f(note.end)))
            tracks[idx, start, note.pitch, 0] = max(end - start, 1)
            tracks[idx, start, note.pitch, 1] = note.velocity
        for control in track.control_changes:
                start = int(np.round(f(control.time)))
                if start >= tracks.shape[1]:
                    break
                tracks[idx, start, control.number, 2] = control.value

    #midi_recon = matrix2midi_with_dynamics(tracks, [0, 0, 0], init_tempo=90)
    #midi_recon.write(f"seg_recon/{song}.mid")

    with open(os.path.join(POP909_MIDI_ROOT, song, f'chord_midi.txt'), 'r') as file:
        chord_annotation = file.readlines()

    chord_matrix = np.zeros(((segmentation[-1][-1] + segmentation[-1][-2]) * 16, 14))
    for chord in chord_annotation:
        start, end, chord = chord.replace('\n', '').split('\t')
        start = int(np.round(f(start)))
        end = int(np.round(f(end)))
        chord_root, bit_map, bass = mir_eval.chord.encode(chord)
        chord = np.concatenate([np.array([chord_root]), np.roll(bit_map, shift=int(chord_root)), np.array([bass])])
        chord_matrix[start: end] = chord
    chord_matrix = chord_matrix[::4]

    song_melody = []
    song_acc = []
    song_chord = []
    song_velocity = []
    song_cc = []

    for (_, length, start) in segmentation:
        song_melody.append(get_ec2_melody(tracks[0, start*16: (start+length)*16, :, 0]))
        song_acc.append(np.max(tracks[1:, start*16: (start+length)*16, :, 0], axis=0))
        song_chord.append(chord_matrix[start*4: (start+length)*4])
        song_velocity.append(np.max(tracks[1:, start*16: (start+length)*16, :, 1], axis=0))
        song_cc.append(tracks[2, start*16: (start+length)*16, :, 2])

    phrase_melody.append(song_melody)
    phrase_acc.append(song_acc)
    phrase_chord.append(song_chord)
    phrase_velocity.append(song_velocity)
    phrase_cc.append(song_cc)

np.savez_compressed('./phrase_data.npz', melody=phrase_melody, acc=phrase_acc, chord=phrase_chord, velocity=phrase_velocity, cc=phrase_cc)