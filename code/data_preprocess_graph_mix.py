import os
import argparse
import numpy as np
np.random.seed(0)
#label mapping is different than data_preprocess_gnn.py
map_label_to_index = {'hateful': 1, 'normal': 0, 'other':2}
edge_feature_file = '../data/users_hate_glove.content'
egde_index_file = '../data/users.edges'

if not os.path.exists('../data/graphmix'):
    os.mkdir('../data/graphmix')

def generate_hate_label_txt():
    with open(edge_feature_file) as f:
        with open('../data/graphmix/label.txt', 'w') as label_file:
            with open('../data/all_labeled_user.txt', 'w') as labeled:
                feature_line = f.read().splitlines()
                for i, feature in enumerate(feature_line):
                    feature = feature.split("\t")
                    label = map_label_to_index[feature[-1]]
                    if label == 0 or label == 1:
                        label_file.write(str(i) + ' ' + str(label) + '\n')
                        labeled.write(str(i)+'\n')

def generate_net_txt():
    with open('../data/users.edges') as f:
        with open('../data/graphmix/net.txt', 'w') as net_file:
            edges_str = f.read().splitlines()
            edge_int = []
            for edge in edges_str:
                net_file.write(edge + ' 1\n')

def generate_train_dev_test():
    with open('../data/graphmix/label.txt') as f:
        users = f.read().splitlines()
        hateful = []
        normal = []
        for user in users:
            uid, label = user.split()
            if label == '1':
                hateful.append(uid)
            else:
                normal.append(uid)
        print(len(hateful), len(normal))
        train_ratio = 464 / len(hateful)
        val_ratio = 50 / len(hateful)
        normal_train = int(len(normal) * train_ratio)
        normal_val = int(len(normal) * val_ratio)
        
        with open('../data/graphmix/train.txt', 'w') as f:
            set_train_hateful = set(np.random.choice(hateful, 464, replace=False))
            set_train_normal = set(np.random.choice(normal, normal_train, replace=False))
            total_trian = list(set_train_hateful.union(set_train_normal))
            np.random.shuffle(total_trian)
            for uid in total_trian:
                f.write(uid + '\n')
            set_val_cand_normal = set(normal) - set_train_normal
            set_val_cand_hate = set(hateful) - set_train_hateful
            set_val_hateful = set(np.random.choice(list(set_val_cand_hate), 50, replace=False))
            set_val_normal = set(np.random.choice(list(set_val_cand_normal), normal_val, replace=False))
            total_val = list(set_val_hateful.union(set_val_cand_normal))
            np.random.shuffle(total_val)
            with open('../data/graphmix/dev.txt', 'w') as f:
                for uid in total_val:
                    f.write(uid+'\n')
            set_test_hateful =  set_val_cand_hate - set_val_hateful
            set_test_normal = set_val_cand_normal - set_val_normal
            total_test = list(set_test_hateful.union(set_test_normal))
            np.random.shuffle(total_test)
            with open('../data/graphmix/test.txt', 'w') as f:
                for uid in total_test:
                    f.write(uid+'\n')
            print('train number:', len(total_trian))
            print('val number:',len(total_val))
            print('test number:',len(total_test))

            


if __name__ == "__main__":
    #process & create the dataset files

    #parser = argparse.ArgumentParser()

    #generate_hate_label_txt()
    #generate_net_txt()
    #generate_train_dev_test()
