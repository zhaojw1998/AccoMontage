import pretty_midi as pyd 
import numpy as np 
import os
from tqdm import tqdm
import pandas as pd 
import platform
import sys
import torch
torch.cuda.current_device()

sys.path.append('./models')
from model import DisentangleVAE
from ptvae import PtvaeDecoder, TextureEncoder
from two_bar_contrastive_model import contrastive_model


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

    melody_set = acc_pool[length][0]
    acc_set = acc_pool[length][1]
    chord_set = acc_pool[length][2]

    weight_key = 'l' + str(last_length) + str(length)

    edge_dict = []
    last_acc_set = acc_pool[last_length][1]
    for item in tqdm(last_acc_set):
        contras_values = contrastive_match(item[np.newaxis, -32:, :], acc_set[:, :32, :], texture_model, contras_model, -1)
        edge_dict.append(contras_values)
    return np.array(edge_dict)
    

data = np.load('./data files/phrase_data0714.npz', allow_pickle=True)
melody = data['melody']
acc = data['acc']
chord = data['chord']

texture_model = TextureEncoder(emb_size=256, hidden_dim=1024, z_dim=256, num_channel=10, for_contrastive=True)
if platform.system() == 'Linux':
    checkpoint = torch.load("/gpfsnyu/scratch/jz4807/model-weights/contrastive_model/params/May  5 00-21-48 2021/texture_model_params049.pt")
else:
    checkpoint = torch.load("./data files/texture_model_params049.pt")
texture_model.load_state_dict(checkpoint)
texture_model.cuda()

contras_model = contrastive_model(emb_size=256, hidden_dim=1024)
if platform.system() == 'Linux':
    contras_model.load_state_dict(torch.load('/gpfsnyu/scratch/jz4807/model-weights/contrastive_model/params/May  5 00-21-48 2021/contrastive_model_params049.pt'))
else:    
    contras_model.load_state_dict(torch.load('./data files/contrastive_model_params049.pt'))
contras_model.cuda()

#length_check = [(8, 8), (4, 4), (8, 4), (4, 8)]
length_check = [(8, 8), (4, 4), (8, 4), (4, 8), (6, 6), (6, 8), (6, 4), (4, 6), (8, 6)]
acc_pool = {}
numpy = []
for lengths in length_check:
    length = lengths[1]
    last_length = lengths[0]
    edge_weights = inference_edge_weights(contras_model, texture_model, length, last_length, melody, acc, chord, acc_pool)
    
    if not os.path.exists('./tmp'):
        os.makedirs('./tmp')
    np.savez_compressed('./tmp/edge_weights' + '_' + str(last_length) + str(length) + '.npz', edge_weights)
    #numpy.append(np.load(file, allow_pickle=True)['arr_0'])

    numpy.append(edge_weights)

np.savez_compressed('./data files/edge_weights_0714.npz', l44=numpy[0], l46=numpy[1], l48=numpy[2], l64=numpy[3], l66=numpy[4], l68=numpy[5], l84=numpy[6], l86=numpy[7], l88=numpy[8])
