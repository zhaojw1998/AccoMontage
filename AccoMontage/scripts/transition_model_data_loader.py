import numpy as np
from tqdm import tqdm
import random
import platform

class dataset(object):
    def __init__(self, data_path='./', batch_size=8, time_res=32):
        song_data = np.load(data_path, allow_pickle=True)['acc']
        self.batch_size = batch_size
        self.time_res = time_res

        self.train_pairs, self.val_pairs, self.snippet_pool = self.find_all_pairs_and_snippet_pool(song_data)

        self.train_batch = None
        self.val_batch = None
        self.train_batch_anchor = None
        self.val_batch_anchor = None
        self.num_epoch = -1

    def song_split(self, matrix):
        """matrix must be quantizded in sixteenth note"""
        window_size = self.time_res #two bars
        hop_size = self.time_res   #one bar
        vector_size = matrix.shape[1]
        start_downbeat = 0
        end_downbeat = matrix.shape[0]//16
        assert(end_downbeat - start_downbeat >= 2)
        splittedMatrix = []
        for idx_T in range(start_downbeat*16, (end_downbeat-1)*16, hop_size):
            sample = matrix[idx_T:idx_T+window_size, :]
            if np.sum(sample) == 0:
                continue    #skip possible blank regions at the head and tail of each song
            splittedMatrix.append(sample)
        return np.array(splittedMatrix)

    def find_all_pairs_and_snippet_pool(self, song_data):
        np.random.seed(0)
        np.random.shuffle(song_data)
        train_data = song_data[: int(len(song_data)*0.95)]
        val_data = song_data[int(len(song_data)*0.95): ]
        train_pairs = []
        val_pairs = []
        snippet_pool = []
        for song in  tqdm(train_data):
            splittedMatrix = self.song_split(song)
            for i in range(splittedMatrix.shape[0] - 1):
                train_pairs.append([splittedMatrix[i], splittedMatrix[i+1]])
                snippet_pool.append(splittedMatrix[i])
            snippet_pool.append(splittedMatrix[-1])
        for song in  tqdm(val_data):
            splittedMatrix = self.song_split(song)
            for i in range(splittedMatrix.shape[0] - 1):
                val_pairs.append([splittedMatrix[i], splittedMatrix[i+1]])
                snippet_pool.append(splittedMatrix[i])
            snippet_pool.append(splittedMatrix[-1])
        return train_pairs, val_pairs, snippet_pool

    def make_batch(self, batch_size):
        print('shuffle dataset')
        random.shuffle(self.train_pairs)
        random.shuffle(self.snippet_pool)
        
        self.train_batch = []
        self.val_batch = []
        self.train_batch_anchor = 0
        self.val_batch_anchor = 0
        self.num_epoch += 1

        for i in tqdm(range(0, len(self.train_pairs)-batch_size, batch_size)):
            batch_pair = np.array(self.train_pairs[i: i+batch_size])
            random_items = np.array(random.sample(self.snippet_pool, batch_size*4)).reshape((batch_size, 4, 32, 128))
            one_batch = np.concatenate((batch_pair, random_items), axis=1)
            #one_batch: batch_size * 6 * 32 * 128
            self.train_batch.append(one_batch)
        if i + batch_size < len(self.train_pairs):
            rest = len(self.train_pairs) - (i + batch_size)
            batch_pair = np.array(self.train_pairs[-rest:])
            random_items = np.array(random.sample(self.snippet_pool, rest*4)).reshape((rest, 4, 32, 128))
            one_batch = np.concatenate((batch_pair, random_items), axis=1)
            self.train_batch.append(one_batch)

        for i in tqdm(range(0, len(self.val_pairs)-batch_size, batch_size)):
            batch_pair = np.array(self.val_pairs[i: i+batch_size])
            random_items = np.array(random.sample(self.snippet_pool, batch_size*4)).reshape((batch_size, 4, 32, 128))
            one_batch = np.concatenate((batch_pair, random_items), axis=1)
            self.val_batch.append(one_batch)
        if i + batch_size < len(self.val_pairs):
            rest = len(self.val_pairs) - (i + batch_size)
            batch_pair = np.array(self.val_pairs[-rest:])
            random_items = np.array(random.sample(self.snippet_pool, rest*4)).reshape((rest, 4, 32, 128))
            one_batch = np.concatenate((batch_pair, random_items), axis=1)
            self.val_batch.append(one_batch)
        print('num_epoch:', self.num_epoch)
        print('shuffle finished')
        print('size of train_batch:', len(self.train_batch))
        print('size of val_batch:', len(self.val_batch))
    
    def get_batch(self, stage='train'):
        if stage == 'train':
            idx = self.train_batch_anchor
            self.train_batch_anchor += 1
            if self.train_batch_anchor == len(self.train_batch):
                self.make_batch(self.batch_size)
            return self.train_batch[idx]
        elif stage == 'val':
            idx = self.val_batch_anchor
            self.val_batch_anchor += 1
            if self.val_batch_anchor == len(self.val_batch):
                self.val_batch_anchor = 0
            return self.val_batch[idx]

    def get_batch_volumn(self, stage='train'):
        if stage == 'train':
            return len(self.train_batch)
        elif stage == 'val':
            return len(self.val_batch)

    def get_epoch(self):
        return self.num_epoch


if __name__ == '__main__':
    import torch
    torch.cuda.current_device()
    import sys
    sys.path.append('AccoMontage')
    from models import contrastive_model, TextureEncoder

    data_Set = dataset('checkpoints/song_data.npz', 1, 32)
    data_Set.make_batch(1)
    init_epoch = 0

    texture_model = TextureEncoder(emb_size=256, hidden_dim=1024, z_dim=256, num_channel=10, for_contrastive=True)
    checkpoint = torch.load("checkpoints/texture_model_params049.pt")
    texture_model.load_state_dict(checkpoint)
    texture_model.cuda()
    texture_model.eval()

    contras_model = contrastive_model(emb_size=256, hidden_dim=1024)   
    contras_model.load_state_dict(torch.load('checkpoints/contrastive_model_params049.pt'))
    contras_model.cuda()
    contras_model.eval()
    
    """ while data_Set.get_epoch() <= 3:
        print(data_Set.get_epoch())
        batch = data_Set.get_batch('train')
        print('train', batch.shape)
        if data_Set.train_batch_anchor == len(data_Set.train_batch):
            #validate
            for i in range(data_Set.get_batch_volumn('val')):
                batch = data_Set.get_batch('val')
                print('validating', batch.shape)"""
        #print(data_Set.get_epoch())
    record = []
    for i in range(data_Set.get_batch_volumn('val')):
        batch = data_Set.get_batch('val')
        #print(batch.shape)
        batch = torch.from_numpy(batch).cuda().float()
        bs, pos_neg, time, roll = batch.shape
        _, batch = texture_model(batch.view(-1, time, roll))
        batch = batch.view(bs, pos_neg, -1)
        similarity = contras_model(batch)
        model_loss = contras_model.contrastive_loss(similarity).cpu().detach().numpy()
        record.append(model_loss)
        record.sort()
    print(record[-100:])