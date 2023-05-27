
import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torch import optim

from transition_model_data_loader import dataset

import sys
sys.path.append('AccoMontage')
from models import contrastive_model, TextureEncoder

args={
    "batch_size": 8,
    "data_path": "checkpoints/song_data.npz",
    'weight_path': "checkpoints/model_master_final.pt",
    "embed_size": 256,
    "hidden_dim": 1024,
    "time_step": 32,
    "n_epochs": 100,
    "lr": 1e-3,
    "decay": 0.99991,
    "log_save": "demo/demo_generate/log",
}
# contrastive optimizer stabalizes at around 10 epochs

class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=last_epoch)

    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(contrastive_model, texture_model, dataset, optimizer, scheduler, loss_recorder, writer):
    batch = dataset.get_batch('train')    #8 * 6 * 8 * 32 * 128
    batch = torch.from_numpy(batch).float().cuda()
    bs, pos_neg, time, roll = batch.shape
    optimizer_2.zero_grad()
    _, batch = texture_model(batch.view(-1, time, roll))
    batch = batch.view(bs, pos_neg, -1)
    

    optimizer.zero_grad()
    similarity = contrastive_model(batch)
    model_loss = contrastive_model.contrastive_loss(similarity)
    model_loss.backward()
    torch.nn.utils.clip_grad_norm_(contrastive_model.parameters(), 1)
    optimizer_2.step()
    optimizer.step()
    loss_recorder.update(model_loss.item())
    scheduler_2.step()
    scheduler.step()

    n_epoch = dataset.get_epoch()
    total_batch = dataset.get_batch_volumn('train')
    current_batch = dataset.train_batch_anchor
    step = current_batch + n_epoch * total_batch

    print('---------------------------Training VAE----------------------------')
    for param in optimizer.param_groups:
        print('lr1: ', param['lr'])
    print('Epoch: [{0}][{1}/{2}]'.format(n_epoch, current_batch, total_batch))
    print('loss: {loss:.5f}'.format(loss=model_loss.item()))
    writer.add_scalar('train_vae/1-loss_total-epoch', loss_recorder.avg, step)
    writer.add_scalar('train_vae/5-learning-rate', param['lr'], step)

def val(contrastive_model, texture_model, dataset, writer, val_loss_recoder):
    loss = val_loss_recoder
    step = 1
    for i in range(dataset.get_batch_volumn('val')):
        batch = dataset.get_batch('val')
        batch = torch.from_numpy(batch).float().cuda()
        bs, pos_neg, time, roll = batch.shape
        _, batch = texture_model(batch.view(-1, time, roll))
        batch = batch.view(bs, pos_neg, -1)
        with torch.no_grad():
            similarity = contrastive_model(batch)
            model_loss = contrastive_model.contrastive_loss(similarity)
            loss.update(model_loss.item())
        n_epoch = dataset.get_epoch()
        total_batch = dataset.get_batch_volumn('val')
        print('----validation----')
        print('Epoch: [{0}][{1}/{2}]'.format(n_epoch, step, total_batch))
        print('loss: {loss:.5f}'.format(loss=model_loss.item()))
        step += 1
    writer.add_scalar('val/loss_total-epoch', loss.avg, n_epoch)

embed_size = args["embed_size"]
hidden_dim = args["hidden_dim"]
weight_path = args["weight_path"]

contrastive_model = contrastive_model(emb_size=embed_size, hidden_dim=hidden_dim).cuda()

texture_model = TextureEncoder(emb_size=256, hidden_dim=1024, z_dim=256, num_channel=10, for_contrastive=True)
checkpoint = torch.load(weight_path)
from collections import OrderedDict
rhy_checkpoint = OrderedDict()
for k, v in checkpoint.items():
    part = k.split('.')[0]
    name = '.'.join(k.split('.')[1:])
    if part == 'rhy_encoder':
        rhy_checkpoint[name] = v
texture_model.load_state_dict(rhy_checkpoint)
texture_model.cuda()

run_time = time.asctime(time.localtime(time.time())).replace(':', '-')
logdir = 'log/' + run_time[4:]
save_dir = 'params/' + run_time[4:]
logdir = os.path.join(args["log_save"], logdir)
save_dir = os.path.join(args["log_save"], save_dir)
batch_size = args['batch_size']
if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

writer = SummaryWriter(logdir)
training_loss_recoder = AverageMeter()
val_loss_recoder = AverageMeter()
dataset = dataset(args['data_path'], batch_size, args['time_step'])
dataset.make_batch(batch_size)

optimizer = optim.Adam(contrastive_model.parameters(), lr=args['lr'])
optimizer_2 = optim.Adam(texture_model.parameters(), lr=1e-4)
scheduler = MinExponentialLR(optimizer, gamma=args['decay'], minimum=1e-5,)
scheduler_2 = MinExponentialLR(optimizer_2, gamma=0.999995, minimum=5e-6,)

while dataset.get_epoch() < args['n_epochs']:
    if dataset.train_batch_anchor == 0:
        contrastive_model.eval()
        val(contrastive_model, texture_model, dataset, writer, val_loss_recoder)
        if (dataset.get_epoch()) % 1 == 0:
            checkpoint = save_dir + '/contrastive_model_params' + str(dataset.get_epoch()).zfill(3) + '.pt'
            torch.save(contrastive_model.cpu().state_dict(), checkpoint)
            contrastive_model.cuda()
            checkpoint = save_dir + '/texture_model_params' + str(dataset.get_epoch()).zfill(3) + '.pt'
            torch.save(texture_model.cpu().state_dict(), checkpoint)
            texture_model.cuda()
            print('Model saved!')
    contrastive_model.train()
    train(contrastive_model, texture_model, dataset, optimizer, scheduler, training_loss_recoder, writer)
    