# cs224w
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



## Dataset
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
