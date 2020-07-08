# AM-GCN
Source code for KDD2020 "[AM-GCN: Adaptive Multi-channel Graph Convolutional Networks](https://arxiv.org/pdf/2007.02265.pdf)"

# Environment Settings 
python == 3.7   
Pytorch == 1.1.0  
Numpy == 1.16.2  
SciPy == 1.3.1  
Networkx == 2.4  
scikit-learn == 0.21.3  

# Usage 
````
python main.py -d dataset -l labelrate
````
dataset: citeseer, uai, acm, BlogCatalog, flickr, coraml
labelrate: 20, 40, 60
e.g.
````
python main.py -d citeseer -l 20
````
# Data

citeseer: [Semi-Supervised Classifcation with Graph Convolutional Networks.](https://github.com/tkipf/pygcn)
uai: [A Unifed Weakly Supervised Framework for Community Detection and Semantic Matching.]
acm: [Heterogeneous Graph Attention Network.](https://github.com/Jhy1993/HAN)
BlogCatalog,flickr: [Co-Embedding Attributed Networks.](https://github.com/mengzaiqiao/CAN)
coraml: [Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via Ranking.](https://github.com/abojchevski/graph2gauss/)

Please unzip the data to use


# Parameter Settings
