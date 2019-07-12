from utils import load_data, accuracy
import numpy as np
import  torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
import time

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, input_, adj):
        support = torch.mm(input_, self.weight) #mm: 行列積
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def __repr__(self):
        return self.__class__.__name__ + "(" \
                         + str(self.in_features) + " -> " \
                        + str(self.out_features) + ")"


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dr_rate):
        super().__init__()
        
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dr_rate = dr_rate
        
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dr_rate, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


def main(device):
    A, X, y, idx_train, idx_val, idx_test = load_data()
    epochs = 200
    lr = 0.01
    weight_decay= 5e-4
    hidden = 16
    dr_rate = 0.5
    n_class = len(np.unique(y.numpy()))

    model = GCN(nfeat=X.shape[1], nhid=hidden, nclass=n_class, dr_rate=dr_rate)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model = model.to(device)
    X = X.to(device)
    A = A.to(device)
    y = y.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    for epoch in range(epochs):
        model.train() #train modeにしている
        optimizer.zero_grad()
        output = model(X, A)
        loss_train = F.nll_loss(output[idx_train], y[idx_train])
        acc_train = accuracy(output[idx_train], y[idx_train])
        loss_train.backward()
        optimizer.step()
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(X, A)
        loss_val = F.nll_loss(output[idx_val], y[idx_val])
        acc_val = accuracy(output[idx_val], y[idx_val])

        print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()))
    
    model.eval()
    output = model(X, A)
    loss_test = F.nll_loss(output[idx_test], y[idx_test])
    acc_test = accuracy(output[idx_test], y[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(device)