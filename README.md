# CS224w
Project repo for Stanford CS224w: Detecting hateful users on Twitter  

## Preprocess data for pytorch geometric (GNN)
Make sure in the data folder there are users.edges, users_hate_all.content, users_hate_glove.content
To create the preprocessed dataset files, create an input folder in data/ then run:
```
cd code
python data_preprocess_gnn.py --feature glove
python data_preprocess_gnn.py --feature all
```
In data/input/, there will be two preprocessed dataset files

## Run GraphSage/GAT on the retweet graph
```
cd code
python fs_graphsage.py #runs with all feature vector & graphsage model
python fs_graphsage.py --feature glove #defaults to all features, 320 dimensions
python fs_graphsage.py --model_type gat #defaults to sage, reproduce graphsage experiment
```
## Run LoNGAE on the retweet graph
```
cd code
git clone https://github.com/vuptran/graph-representation-learning.git
python data_preprocess_LoNGAE.py #get preprocessed data
#Customize model training and metric evaluation
python train_multitask_lpnc.py hateful <gpu_id>
```

## Dataset and original files
Primary dataset: https://www.kaggle.com/manoelribeiro/hateful-users-on-twitter  
Dataset owners' analysis repo: https://github.com/manoelhortaribeiro/HatefulUsersTwitter  
Their GraphSAGE embedding repo: https://github.com/manoelhortaribeiro/GraphSageHatefulUsers  

Their papers:

    "Like Sheep Among Wolves":  Characterizing Hateful Users on Twitter
    Manoel Horta Ribeiro, Pedro H Calais, Yuri A Santos,  Virgílio AF Almeida, Wagner Meira Jr
    MIS2 workshop at WSDM'18
    
    Characterizing and Detecting Hateful Users on Twitter
    Manoel Horta Ribeiro, Pedro H Calais, Yuri A Santos,  Virgílio AF Almeida, Wagner Meira Jr
    ICWSM'18
    
    GraphMix: Regularized Training of Graph Neural Networks for Semi-Supervised Learning
    Verma, V., Qu, M., Lamb, A., Bengio, Y., Kannala, J., & Tang, J. (2019). 
    
    Learning to make predictions on graphs with autoencoders. 
    Tran, P. V. (2018, October). 
    In 2018 IEEE 5th International Conference on Data Science and Advanced Analytics (DSAA) (pp. 237-245). IEEE.
