# AM-GCN
Source code for KDD2020 "[AM-GCN: Adaptive Multi-channel Graph Convolutional Networks](https://arxiv.org/pdf/2007.02265.pdf)"

# Environment Settings 
* python == 3.7   
* Pytorch == 1.1.0  
* Numpy == 1.16.2  
* SciPy == 1.3.1  
* Networkx == 2.4  
* scikit-learn == 0.21.3  

# Usage 
````
python main.py -d dataset -l labelrate
````
* **dataset**: including \[citeseer, uai, acm, BlogCatalog, flickr, coraml\], required.  
* **labelrate**: including \[20, 40, 60\], required.  

e.g.  
````
python main.py -d citeseer -l 20
````

# Data
## Link
* **Citeseer**: [Semi-Supervised Classifcation with Graph Convolutional Networks.](https://github.com/tkipf/pygcn)  
* **UAI2010**: A Unifed Weakly Supervised Framework for Community Detection and Semantic Matching. 
* **ACM**: [Heterogeneous Graph Attention Network.](https://github.com/Jhy1993/HAN)  
* **BlogCatalog,Flickr**: [Co-Embedding Attributed Networks.](https://github.com/mengzaiqiao/CAN)  
* **Coraml**: [Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via Ranking.](https://github.com/abojchevski/graph2gauss/)  

## Usage
Please first **unzip** the data folders and then use. The files in folders are as follows:
````
citeseer/
├─citeseer.edge: edge file.  
├─citeseer.feature: feature file.  
├─citeseer.label: label file.  
├─testL/C.txt: test file. L/C, i.e., Label pre Class, L/C = 20, 40, 60.   
├─trainL/C.txt: train file. L/C, i.e., Label pre Class, L/C = 20, 40, 60.  
└─knn
   └─ck.txt: feature graph file. k = 2~9
````
# Parameter Settings

Recorded in   **./AMGCN/config/[L/C][dataset].ini**  
e.g.   **./AMGCN/config/20citeseer.ini**  

* **Model_setup**: parameters for training AM-GCN, such as nhid1, nhid2, beta, theta... 
* **Data_setting**: dataset setttings, such as paths for input, node numbers, feature dimensions...

