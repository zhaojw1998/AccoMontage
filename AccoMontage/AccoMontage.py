import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np 
import pandas as pd 
import torch
from .acc_utils import split_phrases, computeTIV, chord_shift, cosine, cosine_rhy
from .models import DisentangleVAE
from . import format_converter as cvt
from scipy.interpolate import interp1d
from tqdm import tqdm
import gc


def set_premises(phrase_data_dir, edge_weights_dir, checkpoint_dir, reference_meta_dir, phrase_len=range(1, 17)):
    #load POP909 phrase data
    data = np.load(phrase_data_dir, allow_pickle=True)
    MELODY = data['melody']
    ACC = data['acc']
    CHORD = data['chord']
    VELOCITY = data['velocity']
    CC = data['cc']
    acc_pool = {}
    for length in tqdm(phrase_len):
        (_mel, _acc, _chord, _vel, _cc, _song_reference) = find_by_length(MELODY, ACC, CHORD, VELOCITY, CC, length)
        acc_pool[length] = (_mel, _acc, _chord, _vel, _cc, _song_reference)
    del data, MELODY, ACC, CHORD, VELOCITY, CC
    gc.collect()
    texture_filter = get_texture_filter(acc_pool)   
    #load pre-computed transition score
    edge_weights=np.load(edge_weights_dir, allow_pickle=True)
    #load re-harmonization model
    model = DisentangleVAE.init_model(torch.device('cuda')).cuda()
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint)
    model.eval()
    #load pop909 meta
    reference_check = pd.read_excel(reference_meta_dir)
    return model, acc_pool, reference_check, (edge_weights, texture_filter)


def load_lead_sheet(DEMO_ROOT, SONG_NAME, SEGMENTATION, NOTE_SHIFT, melody_track_ID):
    melody_roll, chord_roll = cvt.leadsheet2matrix(os.path.join(DEMO_ROOT, SONG_NAME, 'lead sheet.mid'), melody_track_ID)
    assert(len(melody_roll == len(chord_roll)))
    if NOTE_SHIFT != 0:
        melody_roll = melody_roll[int(NOTE_SHIFT*4):, :]
        chord_roll = chord_roll[int(NOTE_SHIFT*4):, :]
    if len(melody_roll) % 16 != 0:
        pad_len = (len(melody_roll)//16+1)*16-len(melody_roll)
        melody_roll = np.pad(melody_roll, ((0, pad_len), (0, 0)))
        melody_roll[-pad_len:, -1] = 1
        chord_roll = np.pad(chord_roll, ((0, pad_len), (0, 0)))
        chord_roll[-pad_len:, 0] = -1
        chord_roll[-pad_len:, -1] = -1

    CHORD_TABLE = np.stack([cvt.expand_chord(chord) for chord in chord_roll[::4]], axis=0)
    LEADSHEET = np.concatenate((melody_roll, chord_roll[:, 1: -1]), axis=-1)    #T*142, quantized at 16th
    query_phrases = split_phrases(SEGMENTATION) #[('A', 8, 0), ('A', 8, 8), ('B', 8, 16), ('B', 8, 24)]

    assert len(LEADSHEET)//16 >= sum([item[1] for item in query_phrases]), f'Mismatch in total bar numbers between the MIDI file and the phrase annotation. Detect {len(LEADSHEET)//16} bars in MIDI and {sum([item[1] for item in query_phrases])} bars in the provided phrase annotation.'

    if len(LEADSHEET)//16 > sum([item[1] for item in query_phrases]):
        LEADSHEET = LEADSHEET[:sum([item[1] for item in query_phrases])*16]
        CHORD_TABLE = CHORD_TABLE[:sum([item[1] for item in query_phrases])*4]
    
    return LEADSHEET, CHORD_TABLE, query_phrases


def phrase_selection(LEADSHEET, query_phrases, reference_check, acc_pool, edge_weights, texture_filter=None, PREFILTER=None, SPOTLIGHT=None, randomness=0):
    melody_queries = []
    for item in query_phrases:
        start_bar = item[-1]
        length = item[-2]
        segment = LEADSHEET[start_bar*16: (start_bar+length)*16]
        melody_queries.append(segment)  #melody queries: list of T16*142, segmented by phrases
    print(f'Phrase selection begins: {len(query_phrases)} phrases in total. \n\t Set note density filter: {PREFILTER}.')
    if SPOTLIGHT is not None:
        print(f'\t Refer to {SPOTLIGHT} as much as possible')
    phrase_indice, chord_shift = dp_search(melody_queries, 
                                            query_phrases,
                                            acc_pool, 
                                            edge_weights,
                                            texture_filter,
                                            filter_id = PREFILTER, 
                                            spotlights = ref_spotlight(SPOTLIGHT, reference_check),
                                            randomness = 0.1)
    path = phrase_indice[0]
    shift = chord_shift[0]
    reference_set = []
    for idx_phrase, phrase in enumerate(query_phrases):
        phrase_len = phrase[1]
        song_ref = acc_pool[phrase_len][-1]
        idx_song = song_ref[path[idx_phrase][0]][0]
        pop909_idx = reference_check.iloc[idx_song][0]
        song_name = reference_check.iloc[idx_song][1]
        reference_set.append(f'{idx_phrase}: {str(pop909_idx).zfill(3)}_{song_name}')
    print('Reference pieces:', reference_set)
    return (path, shift)


def find_by_length(melody_data, acc_data, chord_data, velocity_data, cc_data, length):
    """Search from POP909 phrase data for a certain phrase length."""
    melody_record = []
    acc_record = []
    chord_record = []
    velocity_record = []
    cc_record = []
    song_reference = []
    for song_idx in range(acc_data.shape[0]):
        for phrase_idx in range(len(acc_data[song_idx])):
            melody = melody_data[song_idx][phrase_idx]
            if not melody.shape[0] == length * 16:
                continue
            if np.sum(melody[:, :128]) <= 2:
                continue
            melody_record.append(melody)
            acc = acc_data[song_idx][phrase_idx]
            acc_record.append(acc)
            chord = chord_data[song_idx][phrase_idx]
            chord_record.append(chord)
            velocity = velocity_data[song_idx][phrase_idx]
            velocity_record.append(velocity)
            cc = cc_data[song_idx][phrase_idx]
            cc_record.append(cc)
            song_reference.append((song_idx, phrase_idx))
    return np.array(melody_record), np.array(acc_record), np.array(chord_record), np.array(velocity_record), np.array(cc_record), song_reference


def dp_search(query_phrases, seg_query, acc_pool, edge_weights, texture_filter=None, filter_id=None, spotlights=None, randomness=0):
    """Search for texture donors based on dynamic programming.
    * query_phrases: lead sheet in segmented phrases. Shape of each phrase: (T, 142), quantized at 1/4-beat level. This format is defined in R. Yang et al., "Deep music analogy via latent representation disentanglement," ISMIR 2019.
    * seg_query: phrase annotation for the lead sheet. Format of each phrase: (label, length, start). For example, seg_query=[('A', 8, 0), ('A', 8, 8), ('B', 4, 16)].
    * acc_pool: search space for piano texture donors.
    * edge_weights: pre-computed transition scores for texture donor i to i+1.
    * texture_filter: filter on voice number (VN) and rhythmic density (RD).
    * filter_id: specified VN abd RD to filter for the first phrase.
    * spotlights: specified a preference for certain songs and/or artists for the search process.
    * randomness: degree of randomness tobe introduced to the search process.
    """
    seg_query = [item[0] + str(item[1]) for item in seg_query]  #['A8', 'A8', 'B8', 'B8']
    #Searching for phrase 1
    query_length = [query_phrases[i].shape[0]//16 for i in range(len(query_phrases))]
    mel, acc, chord, _, _, song_ref = acc_pool[query_length[0]]
    mel_set = mel
    rhy_set = np.concatenate((np.sum(mel_set[:, :, :128], axis=-1, keepdims=True), mel_set[:, :, 128: 130]), axis=-1)
    query_rhy = np.concatenate((np.sum(query_phrases[0][:, : 128], axis=-1, keepdims=True), query_phrases[0][:, 128: 130]), axis=-1)[np.newaxis, :, :]
    rhythm_result = cosine_rhy(query_rhy+1e-5, rhy_set+1e-5)

    chord_set = chord
    chord_set, num_total, shift_const = chord_shift(chord_set)
    chord_set_TIV = computeTIV(chord_set)
    query_chord = query_phrases[0][:, 130:][::4]
    query_chord_TIV = computeTIV(query_chord)[np.newaxis, :, :]
    chord_score, arg_chord = cosine(query_chord_TIV, chord_set_TIV)

    score = .5*rhythm_result + .5*chord_score
    score += randomness * np.random.normal(0, 1, size=len(score)) #to introduce some randomness
    if spotlights is not None:
        for spot_idx in spotlights:
            for ref_idx, ref_item in enumerate(song_ref):
                    if ref_item[0] == spot_idx: 
                        score[ref_idx] += 1
    if filter_id is not None:
        mask = texture_filter[query_length[0]][0][filter_id[0]] * texture_filter[query_length[0]][1][filter_id[1]] - 1
        score += mask

    path = [[(i, score[i])] for i in range(acc.shape[0])]
    shift = [[shift_const[i]] for i in arg_chord]
    melody_record = np.argmax(mel_set, axis=-1)
    record = []

    #Searching for phrase 2, 3, ...
    for i in tqdm(range(1, len(query_length))):
        mel, acc, chord, _, _, song_ref = acc_pool[query_length[i]]
        weight_key = f"l_{str(query_length[i-1]).zfill(2)}_{str(query_length[i]).zfill(2)}"
        contras_result = edge_weights[weight_key]
        if query_length[i-1] == query_length[i]:
            for j in range(contras_result.shape[0]):
                contras_result[j, j] = -1   #the ith phrase does not transition to itself at i+1
                for k in range(j-1, -1, -1):
                    if song_ref[k][0] != song_ref[j][0]:
                        break
                    contras_result[j, k] = -1   #ith phrase does not transition to its ancestors in the same song.
        if i > 1:
            contras_result = contras_result[[item[-1][1] for item in record]]
        if spotlights is not None:
            for spot_idx in spotlights:
                for ref_idx, ref_item in enumerate(song_ref):
                    if ref_item[0] == spot_idx:
                        contras_result[:, ref_idx] += 1
        mel_set = mel
        rhy_set = np.concatenate((np.sum(mel_set[:, :, :128], axis=-1, keepdims=True), mel_set[:, :, 128: 130]), axis=-1)
        query_rhy = np.concatenate((np.sum(query_phrases[i][:, : 128], axis=-1, keepdims=True), query_phrases[i][:, 128: 130]), axis=-1)[np.newaxis, :, :]
        rhythm_result = cosine_rhy(query_rhy, rhy_set)
        chord_set = chord
        chord_set, num_total, shift_const = chord_shift(chord_set)
        chord_set_TIV = computeTIV(chord_set)
        query_chord = query_phrases[i][:, 130:][::4]
        query_chord_TIV = computeTIV(query_chord)[np.newaxis, :, :]
        chord_score, arg_chord = cosine(query_chord_TIV, chord_set_TIV)
        sim_this_layer = .5*rhythm_result + .5*chord_score
        sim_this_layer += randomness * np.random.normal(0, 1, size=len(sim_this_layer))
        if spotlights is not None:
            for spot_idx in spotlights:
                for ref_idx, ref_item in enumerate(song_ref):
                    if ref_item[0] == spot_idx: 
                        sim_this_layer[ref_idx] += 1
        score_this_layer = .7*contras_result +  .3*np.tile(sim_this_layer[np.newaxis, :], (contras_result.shape[0], 1)) + np.tile(score[:, np.newaxis], (1, contras_result.shape[1]))
        melody_flat =  np.argmax(mel_set, axis=-1)
        if seg_query[i] == seg_query[i-1]:
            melody_pre = melody_record
            matrix = np.matmul(melody_pre, np.transpose(melody_flat, (1, 0))) / (np.linalg.norm(melody_pre, axis=-1)[:, np.newaxis]*(np.linalg.norm(np.transpose(melody_flat, (1, 0)), axis=0))[np.newaxis, :])
            if i == 1:
                for k in range(matrix.shape[1]):
                    matrix[k, :k] = -1
            else:
                for k in range(len(record)):
                    matrix[k, :record[k][-1][1]] = -1
            matrix = (matrix > 0.99) * 1.
            score_this_layer += matrix
        topk = 1
        args = np.argsort(score_this_layer, axis=0)[::-1, :][:topk, :]
        record = []
        for j in range(args.shape[-1]):
            for k in range(args.shape[0]):
                record.append((score_this_layer[args[k, j], j], (args[k, j], j)))
        shift_this_layer = [[shift_const[k]] for k in arg_chord]
        new_path = [path[item[-1][0]] + [(item[-1][1], sim_this_layer[item[-1][1]])] for item in record]
        new_shift = [shift[item[-1][0]] + shift_this_layer[item[-1][1]] for item in record]
        melody_record = melody_flat[[item[-1][1] for item in record]]
        path = new_path
        shift = new_shift
        score = np.array([item[0] for item in record])

    arg = score.argsort()[::-1]
    return [path[arg[i]] for i in range(topk)], [shift[arg[i]] for i in range(topk)]


def re_harmonization(lead_sheet, chord_table, query_phrases, indices, shifts, model, acc_pool, tempo=120):
    """Re-harmonize the accompaniment texture donors and save in MIDI.
    * lead_sheet: the conditional lead sheet. Its melody track will be taken.  Shape: (T, 142), quantized at 1-beat level. This format is defined in R. Yang et al., "Deep music analogy via latent representation disentanglement," ISMIR 2019.
    * chord_table: the conditional chord progression from the lead sheet. Shape: (T', 36), quantized at 1-beat level. This format is defined in Z. Wang et al., "Learning interpretable representation for controllable polyphonic music generation," ISMIR 2020.
    * seg_query: phrase annotation for the lead sheet. Format of each phrase: (label, length, start). For example, seg_query=[('A', 8, 0), ('A', 8, 8), ('B', 4, 16)].
    * indices: the indices of selected texture donor phrases in the acc_pool.
    * shifts: pitch transposition of each selected phrase.
    * acc_pool: search space for piano texture donors.
    * tempo: the tempo to render the piece.
    """
    acc_roll = np.empty((0, 128))
    vel_roll = []
    phrase_mean_vel = []
    cc_roll = np.empty((0, 128))
    #retrive texture donor data of the corrresponding indices from the acc_pool
    for i, idx in enumerate(indices):
        length = query_phrases[i][-2]
        shift = shifts[i]
        # notes
        acc_matrix = np.roll(acc_pool[length][1][idx[0]], shift, axis=-1)
        acc_roll = np.concatenate((acc_roll, acc_matrix), axis=0)
        #MIDI velocity
        vel_matrix = np.roll(acc_pool[length][3][idx[0]], shift, axis=-1)
        phrase_mean_vel.append(np.mean(np.ma.masked_equal(vel_matrix, value=0)))
        vel_roll.append(vel_matrix)
        #MIDI control messages (mainly for pedals)
        cc_matrix = acc_pool[length][4][idx[0]]
        cc_roll = np.concatenate((cc_roll, cc_matrix), axis=0)
    # normalize the scale of velocity across different retrieved phrases
    global_mean_vel = np.mean(np.ma.masked_equal(np.concatenate(vel_roll, axis=0), value=0))
    for i in range(len(vel_roll)):
        vel_roll[i][vel_roll[i] > 0] += (global_mean_vel - phrase_mean_vel[i])
    vel_roll = np.concatenate(vel_roll, axis=0)
    #re-harmonization
    if len(acc_roll) % 32 != 0:
        pad_len = (len(acc_roll)//32+1)*32 - len(acc_roll)
        acc_roll = np.pad(acc_roll, ((0, pad_len), (0, 0)))
        vel_roll = np.pad(vel_roll, ((0, pad_len), (0, 0)))
        cc_roll = np.pad(cc_roll, ((0, pad_len), (0, 0)), mode='constant', constant_values=-1)
        chord_table = np.pad(chord_table, ((0, pad_len//4), (0, 0)))
        chord_table[-pad_len:, 0] = -1
        chord_table[-pad_len:, -1] = -1
    acc_roll = acc_roll.reshape(-1, 32, 128)
    chord_table = chord_table.reshape(-1, 8, 36)
    acc_roll = torch.from_numpy(acc_roll).float().cuda()
    acc_roll = torch.clip(acc_roll, min=0, max=31)
    gt_chord = torch.from_numpy(chord_table).float().cuda()
    est_x = model.inference(acc_roll, gt_chord, sample=False).reshape(-1, 15, 6)
    acc_roll = cvt.grid2pr(est_x)
    #interpolate MIDI velocity
    adapt_vel_roll = np.zeros(vel_roll.shape)
    masked_dyn_matrix = np.ma.masked_equal(vel_roll, value=0)
    mean = np.mean(masked_dyn_matrix, axis=-1)
    onsets = np.nonzero(mean.data)
    dynamic = mean.data[onsets]
    onsets = onsets[0].tolist()
    dynamic = dynamic.tolist()
    if not 0 in onsets:
        onsets = [0] + onsets
        dynamic = [dynamic[0]] + dynamic
    if not len(vel_roll)-1 in onsets:
        onsets = onsets + [len(vel_roll)-1]
        dynamic = dynamic + [dynamic[-1]]
    dyn_curve = interp1d(onsets, dynamic)
    for t, p in zip(*np.nonzero(acc_roll)):
        adapt_vel_roll[t, p] = dyn_curve(t)
    adapt_vel_roll = np.clip(adapt_vel_roll, a_min=0, a_max=127)
    #reconstruct MIDI
    accompaniment = np.stack([acc_roll, adapt_vel_roll, cc_roll], axis=-1)[np.newaxis, :, :, :]
    midi_recon = cvt.matrix2midi_with_dynamics(accompaniment, programs=[0], init_tempo=tempo)
    melody_track = cvt.melody_matrix2data(melody_matrix=lead_sheet[:, :130], tempo=tempo)
    midi_recon.instruments = [melody_track] + midi_recon.instruments
    return midi_recon

def ref_spotlight(ref_name_list, reference_check):
    """convert spotlight song/artist names into the indices of corresponding pieces in the dataset."""
    if ref_name_list is None:
        return None
    check_idx = []
    #POP909 song_id
    for name in ref_name_list:
        line = reference_check[reference_check.song_id == name]
        if not line.empty:
            check_idx.append(line.index)#read by pd, neglect first row, index starts from 0.
    #song name
    for name in ref_name_list:
        line = reference_check[reference_check.name == name]
        if not line.empty:
            check_idx.append(line.index)#read by pd, neglect first row, index starts from 0.
    #artist name
    for name in ref_name_list:
        line = reference_check[reference_check.artist == name]
        if not line.empty:
            check_idx += list(line.index)#read by pd, neglect first row, index starts from 0
    return check_idx


def get_texture_filter(acc_pool):
    """Divide accompaniment texture donors into fifths in terms of voice number (VN) and rhythmic density (RD)."""
    texture_filter = {}
    for key in acc_pool:
        acc_track = acc_pool[key][1]
        # CALCULATE HORIZONTAL DENSITY (rhythmic density)
        onset_positions = (np.sum(acc_track, axis=-1) > 0) * 1.
        HD = np.sum(onset_positions, axis=-1) / acc_track.shape[1]  #(N)
        # CALCULATE VERTICAL DENSITY (voice number)
        beat_positions = acc_track[:, ::4, :]
        downbeat_positions = acc_track[:, ::16, :]
        upbeat_positions = acc_track[:, 2::4, :]

        simu_notes_on_beats = np.sum((beat_positions > 0) * 1., axis=-1)    #N*T
        simu_notes_on_downbeats = np.sum((downbeat_positions > 0) * 1., axis=-1)
        simu_notes_on_upbeats = np.sum((upbeat_positions > 0) * 1., axis=-1)

        VD_beat = np.sum(simu_notes_on_beats, axis=-1) / (np.sum((simu_notes_on_beats > 0) * 1., axis=-1) + 1e-10)
        VD_upbeat = np.sum(simu_notes_on_upbeats, axis=-1) / (np.sum((simu_notes_on_upbeats > 0) * 1., axis=-1) + 1e-10)

        VD = np.max(np.stack((VD_beat, VD_upbeat), axis=-1), axis=-1)
        #get five-equal-divident-points of HD
        dst = np.sort(HD)
        HD_anchors = [dst[len(dst) // 5], dst[len(dst) // 5 * 2], dst[len(dst) // 5 * 3], dst[len(dst) // 5 * 4]]
        HD_Bins = [
            HD < HD_anchors[0],
            (HD >= HD_anchors[0]) * (HD < HD_anchors[1]),
            (HD >= HD_anchors[1]) * (HD < HD_anchors[2]),
            (HD >= HD_anchors[2]) * (HD < HD_anchors[3]),
            HD >= HD_anchors[3]
        ]
        #get five-equal-divident-points of VD
        dst = np.sort(VD)
        VD_anchors = [dst[len(dst) // 5], dst[len(dst) // 5 * 2], dst[len(dst) // 5 * 3], dst[len(dst) // 5 * 4]]
        VD_Bins = [
            VD < VD_anchors[0],
            (VD >= VD_anchors[0]) * (VD < VD_anchors[1]),
            (VD >= VD_anchors[1]) * (VD < VD_anchors[2]),
            (VD >= VD_anchors[2]) * (VD < VD_anchors[3]),
            VD >= VD_anchors[3]
        ]
        texture_filter[key] = (HD_Bins, VD_Bins)    #((5, N), (5, N))
    return texture_filter