import torch
import torch_geometric
from torch_geometric.data import Data, Dataset, InMemoryDataset
import numpy as np
import argparse



def read_edge_index():
    with open(egde_index_file) as f:
        edges_str = f.read().splitlines()
        edge_int = []
        for edge in edges_str:
            node = edge.split(" ")
            edge_int.append(np.array(node, dtype=np.int32))
        np_edges = np.array(edge_int)
    return np.transpose(np_edges)
map_label_to_index = {'hateful': 2, 'normal': 0, 'other':1}
def read_node_feature_file():
    with open(edge_feature_file) as f:
        feature_line = f.read().splitlines()
        features = []
        y = []
        for feature in feature_line:
            feature = feature.split("\t")
            features.append(np.array(feature[1:-1], dtype=np.float32))
            y.append(map_label_to_index[feature[-1]])
        return np.array(y), np.array(features)

class RetweetDataset(InMemoryDataset):
    def __init__(self, root_path, transform=None, pre_transform=None, feature_type='glove'):
        super(RetweetDataset, self).__init__(root_path, transform, pre_transform)
        data_index = 0 if feature_type == 'glove' else 1
        self.data, self.slices = torch.load(self.processed_paths[data_index])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['../data/input/processed_hate_glove_data.dataset', '../data/input/processed_hate_all_data.dataset']
    
    def download(self):
        pass
    
    def process(self):
        #size: (2, E)
        np_edge_index = read_edge_index()
        #size (num_node, 1) (num_node, feature vector size)
        datalist = []
        y, np_node_features = read_node_feature_file()
        torch_edge_index = torch.LongTensor(np_edge_index)
        torch_node_features = torch.FloatTensor(np_node_features)
        torch_y = torch.LongTensor(y)
        data_list = [Data(x=torch_node_features, edge_index = torch_edge_index, y=torch_y)]
        data, slices = self.collate(data_list)
        index = 0 if args.feature == 'glove' else 1
        torch.save((data, slices), self.processed_paths[index])

def construct_dataset(feature='glove'):
    return RetweetDataset("../", feature_type=feature)

def get_labeled_index(user_type='hate', feature_type='glove'):
    edge_feature_file = '../data/users_' + user_type + '_' + feature_type +'.content'
    labeled_hate_index = []
    labeled_normal_index = []
    with open(edge_feature_file) as f:
        feature_line = f.read().splitlines()

        for index, feature in enumerate(feature_line):
            feature = feature.split("\t")
            if (map_label_to_index[feature[-1]] == 2):
                labeled_hate_index.append(index)
            elif (map_label_to_index[feature[-1]] == 0):
                labeled_normal_index.append(index)
    return labeled_hate_index, labeled_normal_index



if __name__ == "__main__":
    #process & create the dataset files

    parser = argparse.ArgumentParser()

    # system
    parser.add_argument("--feature", type=str, default="glove", help="glove | all")
    #no use of user_type for now
    parser.add_argument("--user_type", type=str, default="hate", help="hate | suspend")
    args = parser.parse_args()

    egde_index_file = '../data/users.edges'
    edge_feature_file = '../data/users_' + args.user_type + '_' + args.feature +'.content'
    dataset = RetweetDataset("../", feature_type=args.feature)
    print(dataset[0].num_nodes)