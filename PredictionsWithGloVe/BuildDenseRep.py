# -*- coding: utf-8 -*-

import pickle
import numpy as np
import warnings


"""
Created on Tue Nov 21 16:25:51 2017

@author: eduardofierro

Calls CorpusLoader.py

Builds dense representation of documents using GloVes

"""

def load_gloves(main_data_dir, subdir):
    glove = pickle.load( open(main_data_dir + "/" + subdir + "/" +  "GloVe.p", "rb" ) )
    index_to_word_map = pickle.load( open(main_data_dir + "/" + subdir + "/" +  "index_to_word_map.p", "rb" ) )
    word_to_index_map = pickle.load( open(main_data_dir + "/" + subdir + "/" +  "word_to_index_map.p", "rb" ) )
    
    return glove, index_to_word_map, word_to_index_map
    
def tokenize(string):
    string = string.lower()
    return string.split()

def gloveSingle(sentence, gloves, word_to_index_maps):
    
    tokenized = tokenize(sentence)
    used_vectors = 0
    vector_sum = np.zeros(300)
    for x in tokenized: 
        
        try:
            vector = gloves[word_to_index_maps[x]]
            used_vectors += 1
            vector_sum += vector
        except KeyError:
            # Words not in dictionary
            pass
        
    return vector_sum, used_vectors
    
def gloveDocument(document, gloves, word_to_index_maps):
    
    used_vectors = 0
    vector_sum = np.zeros(300)
    for line in document:
        
        main_vec, count = gloveSingle(line, gloves, word_to_index_maps)
        used_vectors += count
        vector_sum += main_vec

    # Normalized --> Controlling for # of words in the document in the dictionary
    vector_sum = vector_sum/used_vectors
    return vector_sum

def gloveCorpus(corpus, gloves, word_to_index_maps):
    
    print("Building Corpus Vectors...")
    for i, doc in enumerate(corpus):
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # Ignore Warnings. They're due to non-machine readable documents. 
                                            # Will ignore later. 
            doc_vector = gloveDocument(doc, gloves, word_to_index_maps)
        if i == 0: 
            corpus_vectors = doc_vector
        else: 
            corpus_vectors = np.vstack((corpus_vectors, doc_vector))
            
        if i%500 == 0: 
            print("{}/{} advance".format(i, len(corpus)))
    
    print("Corpus Vectors Ready!!")      
    return corpus_vectors