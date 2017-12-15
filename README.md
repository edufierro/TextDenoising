# TextDenoising

Term Project: DS-GA 1013 Optimization-Based Data Analysis

Eduardo Fierro (eff254) & Raúl Delgado (rds491)

## Data scrapping

To download the data, using R (R version 3.3.3), you can run the code `GetData/Scrapping.R`. This code will output the pdf files found for all the bills, as well as an excel and csv file with the relevant information of each document. Each document will be name in a sequence from 1 to N, and uniquely identified with this table. To tranform the data to ".txt" files, run the code `GetData/PDF2TXT.R`. Both of this codes contain instructions to change local directiories at the top of each to point where the files should be saved in each case. Dependencies for these codes are:

```R
require(XML)
require(plyr)
require(bitops)
require(RCurl)
require(stringr)
require(httr)
require(xlsx)
require(R2HTML)
require(readxl)
require(pdftools)
```

Finally, the observations were randomly spit between train/validation/text (70%/15%/15%) using `GetData/DataSplitter.py`. Code runs in python3, with the requirements of numpy and os. 

It's worth noting that the code may require to review the URL of http://sil.gobernacion.gob.mx/portal/AsuntosLegislativos/busquedaBasica, selecting "Iniciativa" and the last "Legislatura" for the scrapper to work properly. 

## GloVe Embeddings

On the subfolder GLove, the code `GloVe.py` generates GloVe embeddings for this project. This codes accepts as parameters the following: 

```python
parser.add_argument('--batchSize', type=int, default=1024, help='input batch size')
parser.add_argument('--main_data_dir', type=str, default='/scratch/eff254/Optimization/Data/', help='input batch size')
parser.add_argument('--enable_minibatch', action='store_true', help='Enables minibatch to the size of --minibatch')
parser.add_argument('--minibatch', type=int, default=400, help='Minibatch (examples to take) for tryouts. Works only if --enable_minibatch')
parser.add_argument('--context_window', type=int, default=5, help='Context Window for Glove Vectors')
parser.add_argument('--top_k', type=int, default=500, help='Vocabulary Size (Top words form)')
parser.add_argument('--learning_rate', type=float, default=1, help='Learning Rate for SGD step on Glove')
parser.add_argument('--embedding_dim', type=int, default=100, help='Dimension of each embedding vector')
parser.add_argument('--num_epochs', type=int, default=2000, help='Number of Epochs')
parser.add_argument('--alpha', type=float, default=0.75, help='GloVe model parameter')
parser.add_argument('--xmax', type=int, default=50, help='GloVe model parameter')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--numpy_random_seed', type=int, default=1234, help='Random Seed when minibatch < len(data)')
parser.add_argument('--min_value', type=int, default=7, help='Min Length of sentences. 99 or more = No trimming. Default=7')
parser.add_argument('--embedding_dim', type=int, default=100, help='Dimension of each embedding vector')
```

This code was run in NYU's Prince (https://wikis.nyu.edu/display/NYUHPC/Clusters+-+Prince). To run the code on prince, see also ``run-glove.sh``. It depends on the following packages: 

```python
import argparse
import re
import collections
import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import nltk.tokenize
import pickle
from itertools import compress
```

The code should output a pickle with a dense Glove numpy matrix, identified by column order using index_to_word_map and word_to_index_map.

## Sparse representation

To build a vocabulary for the sparse representation, ``SparseRepresentation/SparseWords.py``. This code outputs three pickles: vocabulary (a list of counts of the topk words in corpus), index_to_word_map, and word_to_index_map. 

The code runs on python and requires the following packaes: 

```python
import re
import nltk.tokenize
import numpy as np
from collections import Counter
import argparse
import pickle
```
## Baseline predictions and Sparse representation. 

To build the baselines predictions and Sparse representations, the code ``Predictions/SVM_HyperParam_ChenEtAl.py``automatically runs all the possible SVMs with different initializations of $u_0$ for denoising the sparse matrix, hard coded in line 124 of the code, as well as the baseline for the specified vocabulary size. Vocabulary size is restriced for a selection between 50 and 10,000. 

This code takes as parameters the following: 
```python
parser.add_argument('--main_data_dir', type=str, default='/Users/eduardofierro/Google Drive/TercerSemetre/Optimization/Project/Data/', help='Main data dir')
parser.add_argument('--min_value', type=float, default=100, help='Min sentence length to consider (if >99, the command is overwritten')
parser.add_argument('--list_topics', type=str, default="['Educación', 'Campo', 'Sistema Financiero', 'Electoral', 'Derechos Humanos', 'Medio Ambiente', 'Laboral']", help='A list of topics to chose from, as string')
parser.add_argument('--SVM_hyperparam', type=float, default=0.001, help='SVM hyperparam (for all topics)')
parser.add_argument('--SVM_kernel', type=str, default="linear", help="SVM kernel. Default = linear; Must be ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable ")
parser.add_argument('--vocab_size', type=int, default=10000, help="Vocabulary size to use for classification task")
```
It also calls, ```CorpusLoder.py```, ```GetTargets.py```, ```BuildDenseRep.py```, ```BuildSparseRep.py``` and ```DenoiseChenEtAl.py```, all of which should be on the same Predictions folder. 

The code was run in NYU prince, with 4 cpus and 80GB of memory. Once on prince, and sbatch job can be submitted using ```SVM_HyperParam_ChenEtAl-runner.sh```

## Building dense features

To build the dense fuatures you have to tun the folowing code in the save_dense_data.ipynb file:
```python
from Predictions.RunSVM_copy4matrix import *
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from scipy.cluster.vq import whiten

import matplotlib.pyplot as plt

from robust.pcp import pcp
from robust.rpca import *

VocabSizeChecker()
data_dir = 'Data'
min_val = 100
file_type = 'gloves'
vocab_size = 10000

train_features, valid_features = buildFeatures()

topics, train_target, valid_target = buildTargets()
train_features, valid_features, train_target, valid_target = dropMissing(train_features, valid_features, train_target, valid_target)
printTargetBalance(topics, train_target)
```


## Denoising dense representation



