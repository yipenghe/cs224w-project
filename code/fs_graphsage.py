import torch
import torch_geometric
from torch_geometric.data import Data, Dataset, InMemoryDataset, NeighborSampler
from torch_geometric.nn import SAGEConv
from torch import nn
import numpy as np
from data_preprocess_gnn import construct_dataset, get_labeled_index
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, confusion_matrix, recall_score, f1_score, auc, accuracy_score
import torch.nn.functional as F
import random

#get the node dataset graph

class FullySupervisedGraphSageModel(nn.Module):
    def __init__(self, num_features):
        super(FullySupervisedGraphSageModel, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(SAGEConv(num_features, 256))
        self.conv_layers.append(SAGEConv(256, 256))
        self.classify_layer = nn.Linear(256, 2)

    def forward(self, x, data_flow):
        data = data_flow[0]
        x = x[data.n_id]
        x = self.conv_layers[0](x, data.edge_index, size=data.size)
        data = data_flow[1]
        x = self.conv_layers[1](x, data.edge_index, size=data.size)
        scores = self.classify_layer(x)
        return F.log_softmax(scores, dim=1)

def train(loader, data, model, optimizer):
    model.train()
    total_loss = 0
    print("========begin the epoch===========")
    for data_flow in loader(data.train_mask):
        optimizer.zero_grad()
        out = model(data.x, data_flow)
        loss = F.nll_loss(out, data.y[data_flow.n_id], weight=torch.FloatTensor([10, 1]))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_flow.batch_size
    print("========end the epoch==========")
    return total_loss / data.train_mask.sum().item()

def test(loader, data, model, mask):
    model.eval()

    correct = 0
    for data_flow in loader(mask):
        pred = model(data.x, data_flow).max(1)[1]
        correct += pred.eq(data.y[data_flow.n_id]).sum().item()
    return correct / mask.sum().item()

if __name__ == "__main__":
    #process & create the dataset files
    dataset = construct_dataset('glove')
    hate_index, normal_index = get_labeled_index()
    train_index = np.random.choice(hate_index, 400, replace=False)
    test_index = []
    for index in hate_index:
        if index not in train_index:
            test_index.append(index)
    train_normal_index = np.random.choice(normal_index, 4000, replace=False)
    for index in normal_index:
        if index not in train_normal_index:
            test_index.append(index)
    all_train_index = np.concatenate([train_index, train_normal_index])
    data = dataset[0]
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[all_train_index] = 1
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[test_index] = 1
    loader = NeighborSampler(data, size=[25, 10], num_hops=2, batch_size=128, shuffle=True, add_self_loops=True)
    model = FullySupervisedGraphSageModel(data.num_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)
    #loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    for epoch in range(1000):
        loss = train(loader, data, model, optimizer)
        test_acc = test(loader, data, model, data.test_mask)
        print('Epoch: {:02d}, Loss: {:.4f}, Test: {:.4f}'.format(epoch, loss, test_acc))
        if (epoch % 50 == 0):
            print("==============50 EPOCH result======================")
            y_pred = []
            y_true = []
            for data_flow in loader(data.test_mask):
                pred = model(data.x, data_flow).max(1)[1]
                y_pred.extend([1 if v == 1 else 0 for v in pred])
                y_true.extend([1 if v == 1 else 0 for v in data.y[data_flow.n_id]])
            fscore = f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
            recall = recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
            print(confusion_matrix(y_true, y_pred))
            print(fscore, recall)

