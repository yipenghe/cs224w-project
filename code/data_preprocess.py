import torch
import torch_geometric
from torch_geometric.data import Data, Dataset, InMemoryDataset
import numpy as np

egde_index_file = '../data/users.edges'
edge_glove_file = '../data/users_hate_glove.content'
def read_edge_index():
    with open(egde_index_file) as f:
        edges_str = f.read().splitlines()
        edge_int = []
        for edge in edges_str:
            node = edge.split(" ")
            edge_int.append(np.array(node, dtype=np.int32))
        np_edges = np.array(edge_int)
    return np.transpose(np_edges)

def read_node_glove_feature():
    with open(edge_glove_file) as f:
        feature_line = f.read().splitlines()
        features = []
        y = []
        for feature in feature_line:
            feature = feature.split("\t")
            features.append(np.array(feature[1:-1], dtype=np.float32))
            y.append(int(feature[0]))
        return np.array(y), np.array(features)

class RetweetDataset(InMemoryDataset):
    def __init__(self, root_path, transform=None, pre_transform=None):
        super(RetweetDataset, self).__init__(root_path, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['../data/input/processed_glove_data.dataset']
    
    def download(self):
        pass
    
    def process(self):
        #size: (2, E)
        np_edge_index = read_edge_index()
        #size (num_node, 1) (num_node, feature vector size)
        y, np_node_features = read_node_glove_feature()
        torch_edge_index = torch.tensor(np_edge_index, dtype=torch.long)
        torch_node_features = torch.LongTensor(np_node_features)
        torch_y = torch.FloatTensor(y)
        data_list = [Data(x=torch_node_features, edge = torch_edge_index, y=torch_y)]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

dataset=RetweetDataset("../")