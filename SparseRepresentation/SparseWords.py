# -*- coding: utf-8 -*-

import re
import nltk.tokenize
import numpy as np
from collections import Counter
import argparse
import pickle

"""
Created on Fri Nov 24 20:15:52 2017

@author: eduardofierro

Optimization project

Build corpus Sparse matrix representations: 
"""

######## File params ########
parser = argparse.ArgumentParser()
parser.add_argument('--main_data_dir', type=str, default="/Users/eduardofierro/Google Drive/TercerSemetre/Optimization/Project/Data/", help='Main data dictionary')
parser.add_argument('--out_subdir', type=str, default='Sparse', help="Subdir to save vocabulary and maps")
parser.add_argument('--topk', type=int, default=10000, help='Number of Top words to use')
opt = parser.parse_args()
print(opt)

#### Reading functions ####

def readFile(filename, path = opt.main_data_dir): 
    
    with open(path + "/TXTsOriginal/" + filename) as f:
        content = f.read()
    return content

def replaceLineBreaks(text): 
    
    text = re.sub("\n   ", " ", text)
    text = re.sub("\n", " ", text)
    return(text)

def sentenceBreak(text):
    
    text = nltk.sent_tokenize(text)
    return(text)

def loadCorpus(listFiles): 
    
    print("Building Corpus...")
    all_sentences = []
    
    for i, file in enumerate(listFiles): 
        text = readFile(file)
        text = replaceLineBreaks(text)
        text = sentenceBreak(text)
        all_sentences.extend(text)
        
        if i%500 == 0: 
            print("{}/{} advance".format(i, len(listFiles)))
    
    print("Corpus Ready!!")    
    return(all_sentences)

def tokenize(string):
    string = string.lower()
    return string.split()
    
#### Topk ####
    
def buildTopKList(corpus, topk = opt.topk):
    print("Building Vocabulary...")
    mydict = Counter()
    
    for i, line in enumerate(corpus):         
        tok_line = tokenize(line)
        mydict.update(tok_line)
        if i%(round(len(corpus)/10, 0)) ==0:
            print("Advance: {}/{}".format(i, len(corpus)))
            
    mydict = mydict.most_common(topk)
    return [x[0] for x in mydict]

def main(): 
    
    train_examples = np.loadtxt(opt.main_data_dir + "/train.txt", dtype="str")
    corpus_sentences = loadCorpus(train_examples)
    vocabulary = buildTopKList(corpus_sentences)
    index_to_word_map = dict(enumerate(vocabulary))
    word_to_index_map = dict([(index_to_word_map[index], index) for index in index_to_word_map])
    pickle.dump(vocabulary, open(opt.main_data_dir + "/" + opt.out_subdir + "/" + "vocabulary.p", "wb" ) ) 
    pickle.dump(index_to_word_map, open(opt.main_data_dir + "/" + opt.out_subdir + "/" + "index_to_word_map.p", "wb" ) ) 
    pickle.dump(word_to_index_map, open(opt.main_data_dir + "/" + opt.out_subdir + "/" + "word_to_index_map.p", "wb" ) ) 

if __name__ == "__main__" :
    main()    