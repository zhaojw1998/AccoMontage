import numpy as np 
import os
from tqdm import tqdm
import torch
torch.cuda.current_device()

import sys
sys.path.append('AccoMontage')
from models import TextureEncoder, contrastive_model
import warnings
warnings.filterwarnings("ignore")


def find_by_length(melody_data, acc_data, chord_data, length):
    melody_record = []
    acc_record = []
    chord_record = []
    for song_idx in tqdm(range(acc_data.shape[0])):
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
    return np.array(melody_record), np.array(acc_record), np.array(chord_record)


def contrastive_match(left, rights, texture_model, contras_model, num_candidates):
    #left: 1 * time * 128
    #rights: batch * time * 128
    NEG = 6
    batch_size, time, roll_size = rights.shape
    #print(batch_size)
    count = (batch_size // NEG) * NEG
    rights_ = rights[:count].reshape((batch_size // NEG, NEG, time, roll_size))
    left = left[np.newaxis, :, :, :]
    batch_input = np.concatenate((np.repeat(left, rights_.shape[0], axis=0), rights_), axis=1)

    texture_model.eval()
    contras_model.eval()
    consequence = []#np.empty((0, NEG))
    mini_batch = 2
    for i in range(0, batch_input.shape[0] - mini_batch, mini_batch):
        batch = batch_input[i: (i+mini_batch)]
        #lengths = contras_model.get_len_index_tensor(batch)  #8 * 6
        batch = torch.from_numpy(batch).float().cuda()
        #lengths = torch.from_numpy(lengths)
        bs, pos_neg, time, roll = batch.shape
        _, batch = texture_model(batch.view(-1, time, roll))
        batch = batch.view(bs, pos_neg, -1)
        similarity = contras_model(batch)
        consequence.append(similarity.cpu().detach().numpy())
    #print(consequence.shape)
    consequence = np.array(consequence).reshape(-1)
    #print(consequence.shape)

    if (i+mini_batch) < batch_input.shape[0]:
        batch = batch_input[(i + mini_batch): ]
        #lengths = contras_model.get_len_index_tensor(batch)  #8 * 6
        batch = torch.from_numpy(batch).float().cuda()
        #lengths = torch.from_numpy(lengths)
        bs, pos_neg, time, roll = batch.shape
        _, batch = texture_model(batch.view(-1, time, roll))
        batch = batch.view(bs, pos_neg, -1)
        similarity = contras_model(batch).cpu().detach().numpy().reshape(-1)
        consequence = np.concatenate((consequence, similarity))

    if count < batch_size:
        rest = rights[count:].reshape((1, -1, time, roll_size))
        batch = np.concatenate((np.repeat(left, rest.shape[0], axis=0), rest), axis=1)
        #lengths = contras_model.get_len_index_tensor(batch)  #8 * 6
        batch = torch.from_numpy(batch).float().cuda()
        #lengths = torch.from_numpy(lengths)
        bs, pos_neg, time, roll = batch.shape
        _, batch = texture_model(batch.view(-1, time, roll))
        batch = batch.view(bs, pos_neg, -1)
        similarity = contras_model(batch).cpu().detach().numpy().reshape(-1)
        consequence = np.concatenate((consequence, similarity))
    #print(batch_size, consequence.shape)
    if num_candidates == -1:
        #argmax = np.argsort(consequence)[::-1]
        return consequence#, argmax
    else:
        argmax = np.argsort(consequence)[::-1]
        #result = [consequence[i] for i in argmax]
        #print(result, argmax[:num_candidates])
        return consequence, argmax[:num_candidates]


def inference_edge_weights(contras_model, texture_model, length, last_length, melody_data, acc_data, chord_data, acc_pool):
    if not length in acc_pool:
        (mel, acc, chord) = find_by_length(melody_data, acc_data, chord_data, length)
        acc_pool[length] = (mel, acc, chord)
    if not last_length in acc_pool:
        (mel, acc, chord) = find_by_length(melody_data, acc_data, chord_data, last_length)
        acc_pool[last_length] = (mel, acc, chord)

   # melody_set = acc_pool[length][0]
    acc_set = acc_pool[length][1]
    #chord_set = acc_pool[length][2]

    edge_dict = []
    last_acc_set = acc_pool[last_length][1]
    for item in tqdm(last_acc_set):
        if len(item) < 32:
            item = np.pad(item, ((32-len(item), 0), (0, 0)))
        if acc_set.shape[1] < 32:
            acc_set = np.pad(acc_set, ((0, 0), (0, 32-acc_set.shape[1]), (0, 0)))
        contras_values = contrastive_match(item[np.newaxis, -32:, :], acc_set[:, :32, :], texture_model, contras_model, -1)
        edge_dict.append(contras_values)
    return np.array(edge_dict)
    

data = np.load('checkpoints/phrase_data.npz', allow_pickle=True)
melody = data['melody']
acc = data['acc']
chord = data['chord']

texture_model = TextureEncoder(emb_size=256, hidden_dim=1024, z_dim=256, num_channel=10, for_contrastive=True)
checkpoint = torch.load("checkpoints/texture_model_params049.pt")
texture_model.load_state_dict(checkpoint)
texture_model.cuda()

contras_model = contrastive_model(emb_size=256, hidden_dim=1024)
contras_model.load_state_dict(torch.load('checkpoints/contrastive_model_params049.pt'))
contras_model.cuda()

for l1 in range(1, 17):
    for l2 in range(1, 17):
        length = l2
        last_length = l1
        edge_weights = inference_edge_weights(contras_model, texture_model, length, last_length, melody, acc, chord, {})
        
        if not os.path.exists('./tmp'):
            os.makedirs('./tmp')
        np.savez_compressed('./tmp/edge_weights' + '_' + str(last_length) + '_' + str(length) + '.npz', edge_weights)
