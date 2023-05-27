import torch
from torch import nn

class contrastive_model(nn.Module):
    def __init__(self, emb_size=256, hidden_dim=1024):
        """input: ((batch * 6) * (1024*2))"""
        super(contrastive_model, self).__init__()
        #self.in_linear = nn.Linear(1024*2, emb_size)
        self.out_linear_left = nn.Linear(hidden_dim * 2, emb_size)
        self.out_linear_right = nn.Linear(hidden_dim * 2, emb_size)

        self.emb_size = emb_size
        self.hidden_dim = hidden_dim

        self.dropout = nn.Dropout(p=0)

        self.cosine = nn.CosineSimilarity(dim=-1)
        self.loss = nn.Softmax(dim=-1)

    def contrastive_loss(self, similarity):
        return 1 - torch.mean(self.loss(similarity)[:, 0])

    def forward(self, batch):
        """input: (batch * 6 * (1024*2))"""
        batch_size, pos_neg, feature_dim = batch.shape
        #batch = self.in_linear(batch) #(batch_size * pos_neg_size) * phrase_length * emb_size
        left = self.dropout(self.out_linear_left(batch[:, 0: 1, :]))  #batch * 1 * emb_size
        right = self.dropout(self.out_linear_right(batch[:, 1:, :]))  #batch * 5 * emb_size
        similarity = self.cosine(left.expand(right.shape), right)   #batch * 5
        return similarity

