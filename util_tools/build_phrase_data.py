import pretty_midi as pyd 
import numpy as np 
import os
from tqdm import tqdm
import pandas as pd 

import sys
sys.path.append('./util_tools')
import format_converter as cvt

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

def get_ec2_melody(pr_matrix, start_beat, end_beat):
    hold_pitch = 128
    rest_pitch = 129
    piano_roll = np.zeros(((end_beat-start_beat)*4, 130))
    #piano_roll[:, rest_pitch] = 1.
    #chroma = np.zeros(((end_beat-start_beat)*4, 12))
    for item in pr_matrix: #take one role
        if item[0] >= start_beat and item[0] < end_beat:
            t = (item[0] - start_beat) * 4 + item[1]
            pitch = item[-2]
            duration = (item[3] - item[0]) * 4 + (item[4] - item[1])
            piano_roll[t, pitch] = 1
            for i in range(1, duration):
                        if t+i >= piano_roll.shape[0]:
                            break
                        piano_roll[t+i, hold_pitch] = 1
            for i in range(piano_roll.shape[0]):
                if np.sum(piano_roll[i]) == 0:
                    piano_roll[i, rest_pitch] = 1
    return piano_roll

def get_prMatrix(matrix, start_beat, end_beat):
    pr_matrix = np.zeros(((end_beat-start_beat)*4, 128))
    for item in matrix:
        if item[0] >= start_beat and item[0] < end_beat:
            t = (item[0] - start_beat) * 4 + item[1]
            pitch = item[-2]
            duration = (item[3] - item[0]) * 4 + (item[4] - item[1])
            pr_matrix[t, pitch] = duration
    #if np.sum(pr_matrix) == 0:
        #print(start_beat, end_beat)
    return pr_matrix

def get_chord(song_id, line, matrix, start_beat, end_beat):
    pr_chord = np.zeros((end_beat-start_beat, 14))
    pr_chord[:, 0] = -1
    pr_chord[:, -1] = -1
    start = max(start_beat, 0)
    end = min(end_beat, matrix.shape[0])
    if  start_beat < 0 or end_beat > matrix.shape[0]:
        print(song_id, start_beat, end_beat, matrix.shape[0])
    try:
        pr_chord[0+start-start_beat: pr_chord.shape[0]+end-end_beat] = matrix[start: end, :]
    except ValueError:
        print(song_id, start_beat, end_beat)
    return pr_chord

def render(song_id, line, generate_midi=False):
    data=np.load('./data files/POP909 4bin quntization/POP09-PIANOROLL-4-bin-quantization/'+str(song_id).zfill(3)+'.npz')
    chord = data['chord']
    melody = data['melody']
    bridge = data['bridge']
    piano = data['piano']
    ec2vae_ensemble = np.empty((0, 142))
    melody_record = []
    pr_matrix_mix = []#np.empty((0, 128))
    chord_record = []
    chord_ensemble = np.empty((0, 14))
    for item in line:
        start = item[1][0]
        end = item[1][1]
        if start == end:
            print(song_id, 'empty phrase')
            continue
        #if item[0].isupper():
        #pr_melody = get_prMatrix(melody, start, end)
        ec2_melody = get_ec2_melody(melody, start, end)
        pr_bridge = get_prMatrix(bridge, start, end)
        pr_acc = get_prMatrix(piano, start, end)
        pr_matrix = pr_acc + pr_bridge #+ pr_melody
        chord_phrase = get_chord(song_id, line, chord, start, end)
        #print(pr_melody.shape, pr_matrix.shape, chord_phrase.shape)
        if not np.sum(pr_matrix) == 0:
            melody_record.append(ec2_melody)
            pr_matrix_mix.append(pr_matrix)
            chord_record.append(chord_phrase)

    if generate_midi:
        recon_poly = cvt.accompany_matrix2data(pr_matrix_mix.transpose(), 80)
        midi = pyd.PrettyMIDI()
        midi.instruments.append(recon_poly)
        if not os.path.exists('./data files/POP909 4bin quntization/POP-909_recon'):
            os.makedirs('./data files/POP909 4bin quntization/POP-909_recon')
        midi.write(os.path.join('./data files/POP909 4bin quntization/POP-909_recon/', str(song_id) + '_accompaniment.mid'))

    return melody_record, pr_matrix_mix, chord_record

def build_phrase_dataset():
    segmentation_root = './data files/POP909 hierarchical-structure-analysis'
    df = pd.read_excel('./data files/POP909 4bin quntization//index.xlsx')
    full_melody = []
    full_acc = []
    full_chord = []
    for song_id in tqdm(range(1, 910)):
        #print(song_id)
        meta_data = df[df.song_id == song_id]
        num_beats = meta_data.num_beats_per_measure.values[0]
        num_quavers = meta_data.num_quavers_per_beat.values[0]
        if int(num_beats) == 3 or int(num_quavers) == 3:
            #print('triple_beat')
            continue
        try:
            with open(os.path.join(segmentation_root, str(song_id).zfill(3)+'/segsections.txt'), 'r') as f:
                info = f.readlines()[0]
        except:
            print(song_id)
            continue
        reference = split_phrases(info)
        for idx_m in range(len(reference)):
            if reference[idx_m][0].isupper():
                break   # skip intro part
        reference = [(item[0], item[1], item[2] - reference[idx_m][2]) for item in reference]
        for i in range(len(reference)):
            if reference[i][0].isupper():
                melody_start_measure = sum([item[1] for item in reference[:i]])
                break
        with open(os.path.join(segmentation_root, str(song_id).zfill(3)+'/melody.txt'), 'r') as f:
            intro_length = f.readlines()[0].split(' ')
        if int(intro_length[0]) == 0:
            melody_start_note = int(intro_length[1])
        else:
            melody_start_note = 0
        beat_shift = melody_start_measure*4 - melody_start_note//4
        data=np.load('./data files/POP909 4bin quntization/POP09-PIANOROLL-4-bin-quantization/'+str(song_id).zfill(3)+'.npz')
        melody = data['melody']
        melody_start_beat = melody[0, 0] + beat_shift
        phrases_count_by_beat = []
        for phrase in reference:
            label = phrase[0]
            start = phrase[2]*4 + melody_start_beat
            end = start + phrase[1]*4
            phrases_count_by_beat.append([label, (start, end)])
        #return phrases_count_by_beat
        
        melody, acc, chord = render(song_id, phrases_count_by_beat, generate_midi=False)
        full_melody.append(melody)
        full_acc.append(acc)
        full_chord.append(chord)
    np.savez_compressed('./data files/phrase_data_.npz', melody=full_melody, acc=full_acc, chord=full_chord)

def add_phrases(leadsheet, accompany, phrase_sections, jump=0):
    melody, chord = leadsheet.instruments[0], leadsheet.instruments[1]
    downbeats = leadsheet.get_downbeats()
    melody_matrix = cvt.melody_data2matrix(melody, downbeats)[int(jump*16):]
    chord_table = cvt.chord_data2matrix(chord, downbeats, resolution='beat', chord_expand=False)[int(jump*4):]
    downbeats = accompany.get_downbeats()
    accompany_matrix = cvt.accompany_data2matrix(accompany.instruments[1], downbeats)[int(jump*16):]
    count = 0
    melodys = []
    chords = []
    accs = []
    for label in phrase_sections:
        name = label[0]
        length = int(label[1:])
        if name.isupper():
            melody_cut = melody_matrix[count*16: (count+length)*16, :]
            chord_cut = chord_table[count*4: (count+length)*4, :]
            acc_cut = accompany_matrix[count*16: (count+length)*16, :]
            melodys.append(melody_cut)
            chords.append(chord_cut)
            accs.append(acc_cut)
        count += length
    return melodys, chords, accs


def build_song_dataset():
    segmentation_root = './data files/POP909 hierarchical-structure-analysis'
    df = pd.read_excel('./data files/POP909 4bin quntization//index.xlsx')
    full_melody = []
    full_acc = []
    full_chord = []
    for song_id in tqdm(range(1, 910)):
        #print(song_id)
        meta_data = df[df.song_id == song_id]
        num_beats = meta_data.num_beats_per_measure.values[0]
        num_quavers = meta_data.num_quavers_per_beat.values[0]
        if int(num_beats) == 3 or int(num_quavers) == 3:
            #print('triple_beat')
            continue
        try:
            with open(os.path.join(segmentation_root, str(song_id).zfill(3)+'/segsections.txt'), 'r') as f:
                info = f.readlines()[0]
        except:
            print(song_id)
            continue
        reference = split_phrases(info)
        for idx_m in range(len(reference)):
            if reference[idx_m][0].isupper():
                break   # skip intro part
        reference = [(item[0], item[1], item[2] - reference[idx_m][2]) for item in reference]
        for i in range(len(reference)):
            if reference[i][0].isupper():
                melody_start_measure = sum([item[1] for item in reference[:i]])
                break
        with open(os.path.join(segmentation_root, str(song_id).zfill(3)+'/melody.txt'), 'r') as f:
            intro_length = f.readlines()[0].split(' ')
        if int(intro_length[0]) == 0:
            melody_start_note = int(intro_length[1])
        else:
            melody_start_note = 0
        beat_shift = melody_start_measure*4 - melody_start_note//4
        data=np.load('./data files/POP909 4bin quntization/POP09-PIANOROLL-4-bin-quantization/'+str(song_id).zfill(3)+'.npz')
        melody = data['melody']
        chord = data['chord']
        bridge = data['bridge']
        piano = data['piano']
        melody_start_beat = melody[0, 0] + beat_shift
        phrases_count_by_beat = []

        start = reference[0][2]*4 + melody_start_beat
        end = reference[-1][2]*4 + melody_start_beat + reference[-1][1]*4

        ec2_melody = get_ec2_melody(melody, start, end)
        pr_bridge = get_prMatrix(bridge, start, end)
        pr_acc = get_prMatrix(piano, start, end)
        pr_matrix = pr_acc + pr_bridge
        chord_track = get_chord(song_id, 1, chord, start, end)

        full_melody.append(ec2_melody)
        full_acc.append(pr_matrix)
        full_chord.append(chord_track)

        if np.random.rand() > 0.9:
            recon_poly = cvt.accompany_matrix2data(pr_matrix.transpose(), 80)
            midi = pyd.PrettyMIDI()
            midi.instruments.append(recon_poly)
            midi.write(os.path.join('./test_midi.mid'))
    np.savez_compressed('./data files/song_data.npz', melody=full_melody, acc=full_acc, chord=full_chord)


if __name__ == '__main__':
    build_phrase_dataset()
    #build_song_dataset()
    
    """#add songs beyond pop 909
    data = np.load('./data files/phrase_data.npz', allow_pickle=True)
    melody = list(data['melody'])
    acc = list(data['acc'])
    chord = list(data['chord'])
    
    lead_sheet = pyd.PrettyMIDI('./data files/extra songs/La Marseillaise lead_sheet.mid')
    accompany = pyd.PrettyMIDI('./data files/extra songs/La Marseillaise.mid')
    melodys, chords, accs = add_phrases(lead_sheet, accompany, ['A4', 'B8', 'C6', 'D4', 'E6', 'D4', 'E6'], jump=1)
    melody.append(melodys)
    chord.append(chords)
    acc.append(accs)

    lead_sheet = pyd.PrettyMIDI('./data files/extra songs/Internationale v1 lead_sheet.mid')
    accompany = pyd.PrettyMIDI('./data files/extra songs/Internationale v1.mid')
    melodys, chords, accs = add_phrases(lead_sheet, accompany, ['A8', 'B8', 'B8', 'C8'], jump=1)
    melody.append(melodys)
    chord.append(chords)
    acc.append(accs)

    lead_sheet = pyd.PrettyMIDI('./data files/extra songs/Internationale v2 lead_sheet.mid')
    accompany = pyd.PrettyMIDI('./data files/extra songs/Internationale v2.mid')
    melodys, chords, accs = add_phrases(lead_sheet, accompany, ['A8', 'B8', 'B8', 'C8', 'A8', 'B8', 'B8', 'C8'], jump=0.25)
    melody.append(melodys)
    chord.append(chords)
    acc.append(accs)

    np.savez_compressed('./data files/phrase_data0714.npz', melody=melody, acc=acc, chord=chord)
    """