import torch
import torch_geometric
from torch_geometric.data import Data, Dataset, InMemoryDataset, NeighborSampler
from torch_geometric.nn import SAGEConv, GATConv
from torch import nn
import numpy as np
from data_preprocess_gnn import construct_dataset, get_labeled_index
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, confusion_matrix, recall_score, f1_score, auc, accuracy_score, precision_score
import torch.nn.functional as F
import random
from torch.nn import init
import argparse


#get the node dataset graph

class FullySupervisedGraphSageModel(nn.Module):
    def __init__(self, num_features):
        super(FullySupervisedGraphSageModel, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(SAGEConv(num_features, 256))
        #self.conv_layers.append(SAGEConv(256, 256))
        self.classify_layer = nn.Linear(256, 3)
        init.xavier_uniform_(self.classify_layer.weight)
        init.xavier_uniform_(self.conv_layers[0].weight)

    def forward(self, x, data_flow):
        data = data_flow[0]
        x = x[data.n_id]
        x = self.conv_layers[0](x, data.edge_index, size=data.size)
        # data = data_flow[1]
        # x = self.conv_layers[1](x, data.edge_index, size=data.size)
        scores = self.classify_layer(x)
        return F.log_softmax(scores, dim=1)

class FullySupervisedGATModel(nn.Module):
    def __init__(self, num_features):
        super(FullySupervisedGATModel, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GATConv(num_features, 256))
        #self.conv_layers.append(SAGEConv(256, 256))
        self.classify_layer = nn.Linear(256, 3)
        init.xavier_uniform_(self.classify_layer.weight)
        init.xavier_uniform_(self.conv_layers[0].weight)


    def forward(self, x, data_flow):
        block = data_flow[0]
        x = x[block.n_id]
        x = self.conv_layers[0]((x, x[block.res_n_id].squeeze()), block.edge_index, size=block.size)
        # data = data_flow[1]
        # x = self.conv_layers[1](x, data.edge_index, size=data.size)
        scores = self.classify_layer(x)
        return F.log_softmax(scores, dim=1)

def train(loader, data, model, optimizer):
    model.train()
    total_loss = 0
    for data_flow in loader(data.train_mask):
        optimizer.zero_grad()
        out = model(data.x, data_flow)
        loss = F.nll_loss(out, data.y[data_flow.n_id], weight=torch.FloatTensor([1, 0 , 10]))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_flow.batch_size
    return total_loss / data.train_mask.sum().item()

def test(loader, data, model, mask):
    model.eval()
    y_pred = []
    y_true = []
    correct = 0
    for data_flow in loader(mask):
        pred = model(data.x, data_flow).max(1)[1]
        correct += pred.eq(data.y[data_flow.n_id]).sum().item()
        y_pred.extend([1 if v == 2 else 0 for v in pred])
        y_true.extend([1 if v == 2 else 0 for v in data.y[data_flow.n_id]])
    return correct / mask.sum().item(), y_pred, y_true

if __name__ == "__main__":
    #process & create the dataset files
    parser = argparse.ArgumentParser()

    # system
    parser.add_argument("--feature", type=str, default="all", help="glove | all")
    #no use of user_type for now
    parser.add_argument("--user_type", type=str, default="hate", help="hate | suspend")
    parser.add_argument("--model_type", type=str, default="sage", help="sage | gat")
    parser.add_argument("--epoch", type=int, default=201)
    args = parser.parse_args()
    assert(args.feature in ['glove', 'all'])
    assert(args.user_type in ['hate', 'suspend'])
    assert(args.model_type in ['sage', 'gat'])
    print("====information of experiment====")
    print("FEATURE: ", args.feature, "classification_type:", args.user_type, "MODEL:", args.model_type)
    print("====end information of experiment====")
    dataset = construct_dataset(args.feature)
    model_type = args.model_type
    hate_index, normal_index = get_labeled_index(feature_type=args.feature)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    y_all = [2] * len(hate_index)
    y_normal = [0] * len(normal_index)
    y_all.extend(y_normal)
    all_index = []
    all_index.extend(hate_index)
    all_index.extend(normal_index)
    recall_test = []
    accuracy_test = []
    fscore_test = []
    precision_test = []
    all_index = np.array(all_index)
    trail = 0
    for train_i, test_i in skf.split(all_index, y_all):
        print("========begin trail {:01d}===========".format(trail))
        all_train_index = all_index[train_i]
        test_index = all_index[test_i]
        data = dataset[0]
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[all_train_index] = 1
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[test_index] = 1
        loader = NeighborSampler(data, size=[25], num_hops=1, batch_size=128, shuffle=True, add_self_loops=True)
        if model_type == 'sage':
            model = FullySupervisedGraphSageModel(data.num_features)
        else:
            model = FullySupervisedGATModel(data.num_features)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
        #loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        for epoch in range(args.epoch):
            loss = train(loader, data, model, optimizer)
            #test_acc = test(loader, data, model, data.test_mask)
            print('Trail: {:01d}, Epoch: {:02d}, Loss: {:.4f}'.format(trail, epoch, loss))
            if (epoch % 50 == 0):                
                test_acc, y_pred, y_true = test(loader, data, model, data.test_mask)
                fscore = f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
                recall = recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
                precision = precision_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
                print(confusion_matrix(y_true, y_pred))
                print("Fscore:",fscore, "Recall:", recall, "Precision:", precision, "Test:", test_acc)
        model.eval()
        y_pred = []
        y_true = []
        correct = 0
        for data_flow in loader(data.test_mask):
            pred = model(data.x, data_flow).max(1)[1]
            correct += pred.eq(data.y[data_flow.n_id]).sum().item()
            y_pred.extend([1 if v == 2 else 0 for v in pred])
            y_true.extend([1 if v == 2 else 0 for v in data.y[data_flow.n_id]])
        test_acc = correct / data.test_mask.sum().item()
        fscore = f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
        recall = recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
        precision = precision_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
        accuracy_test.append(test_acc)
        recall_test.append(recall)
        fscore_test.append(fscore)
        precision_test.append(precision)
        trail+=1
        print("========end this trail==========")
    accuracy_test = np.array(accuracy_test)
    recall_test = np.array(recall_test)
    fscore_test = np.array(fscore_test)
    precision_test = np.array(precision_test)
    print("avg Accuracy   %0.4f +-  %0.4f" % (accuracy_test.mean(), accuracy_test.std()))
    print("avg Recall    %0.4f +-  %0.4f" % (recall_test.mean(), recall_test.std()))
    print("avg Precision    %0.4f +-  %0.4f" % (precision_test.mean(), precision_test.std()))
    print("avg Fscore    %0.4f +-  %0.4f" % (fscore_test.mean(), fscore_test.std()))


