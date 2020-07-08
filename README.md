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
**dataset**: citeseer, uai, acm, BlogCatalog, flickr, coraml  
**labelrate**: 20, 40, 60  
e.g.  
````
python main.py -d citeseer -l 20
````
# Data

# Link
**Citeseer**: [Semi-Supervised Classifcation with Graph Convolutional Networks.](https://github.com/tkipf/pygcn)  
**UAI2010**: A Unifed Weakly Supervised Framework for Community Detection and Semantic Matching. 
**ACM**: [Heterogeneous Graph Attention Network.](https://github.com/Jhy1993/HAN)  
**BlogCatalog,Flickr**: [Co-Embedding Attributed Networks.](https://github.com/mengzaiqiao/CAN)  
**Coraml**: [Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via Ranking.](https://github.com/abojchevski/graph2gauss/)  

# Use
Please unzip the data to use.

D:.
│  citeseer.edge
│  citeseer.feature
│  citeseer.label
│  test.txt
│  test20.txt
│  test40.txt
│  test60.txt
│  train20.txt
│  train40.txt
│  train60.txt
│  
└─knn
        c2.txt
        c3.txt
        c4.txt
        c5.txt
        c6.txt
        c7.txt
        c8.txt
        c9.txt
        



# Parameter Settings
