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

def load_sparse(main_data_dir, subdir):
    vocabylary = pickle.load( open(main_data_dir + "/" + subdir + "/" +  "vocabulary.p", "rb" ) )
    index_to_word_map = pickle.load( open(main_data_dir + "/" + subdir + "/" +  "index_to_word_map.p", "rb" ) )
    word_to_index_map = pickle.load( open(main_data_dir + "/" + subdir + "/" +  "word_to_index_map.p", "rb" ) )
    
    return vocabylary, index_to_word_map, word_to_index_map

def tokenize(string):
    string = string.lower()
    return string.split()

def sparseDocument(document, vocabylary, word_to_index_maps):
    
    matrix_row = np.zeros(len(vocabylary))
    
    for sentence in document:
    	tokenized = tokenize(sentence)
    	for word in tokenized: 

    		try: 
    			index = word_to_index_maps[word]
    			matrix_row[index] += 1
    		except KeyError:
    			# Words not in dictionary
    			pass

    return matrix_row

def sparseCorpus(corpus, vocabylary, word_to_index_maps):
    
    print("Building Corpus Sparse Representation...")
    for i, doc in enumerate(corpus):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # Ignore Warnings. They're due to non-machine readable documents. 
                                            # Will ignore later. 

            doc_vector =  sparseDocument(doc, vocabylary, word_to_index_maps)
        if i == 0: 
            corpus_vectors = doc_vector
        else: 
            corpus_vectors = np.vstack((corpus_vectors, doc_vector))  

        if i%500 == 0: 
            print("{}/{} advance".format(i, len(corpus)))  	

    
    print("Corpus Vectors Ready!!")      
    return corpus_vectors