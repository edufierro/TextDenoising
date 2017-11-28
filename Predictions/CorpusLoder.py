 # -*- coding: utf-8 -*-

from itertools import compress
import re
import nltk.tokenize 

"""
Created on Tue Nov 21 16:04:38 2017

@author: eduardofierro

Load corpus
"""

def readFile(filename, path): 
    
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

def trim_corpus(corpus, min_len):
    
    mask = [len(line.split(' ')) >= min_len for line in corpus]
    corpus = list(compress(corpus, mask)) 
    
    return (corpus) 

def loadCorpus(listFiles, path, min_value=100): 
    
    '''Same as loadCorpus() on GloVe.py files
       but appends list, instead of extending.
       Incorporates trime_corpus on this samea function.'''
    
    print("Building Corpus...")
    all_sentences = []
    
    for i, file in enumerate(listFiles): 
        text = readFile(file, path=path)
        text = replaceLineBreaks(text)
        text = sentenceBreak(text)
        
        if min_value<99: 
            text = trim_corpus(text, min_value) 
        
        all_sentences.append(text)
        
        if i%500 == 0: 
            print("{}/{} advance".format(i, len(listFiles)))
    
    print("Corpus Ready!!")    
    return(all_sentences)  