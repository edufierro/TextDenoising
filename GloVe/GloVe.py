# -*- coding: utf-8 -*-

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

"""
Created on Sun Nov  5 13:03:11 2017

@author: eduardofierro

Create GloVe vectors

Code based on NLP: Homewwork 1
"""

######## File params########

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1024, help='input batch size')
parser.add_argument('--main_data_dir', type=str, default='/scratch/eff254/Optimization/Data/', help='input batch size')
parser.add_argument('--minibatch', type=int, default=400, help='Minibatch (examples to take) for tryouts.')
parser.add_argument('--context_window', type=int, default=5, help='Context Window for Glove Vectors')
parser.add_argument('--top_k', type=int, default=500, help='Vocabulary Size (Top words form)')
parser.add_argument('--learning_rate', type=float, default=1, help='Learning Rate for SGD step on Glove')
parser.add_argument('--embedding_dim', type=int, default=100, help='Dimension of each embedding vector')
parser.add_argument('--num_epochs', type=int, default=2000, help='Number of Epochs')
parser.add_argument('--alpha', type=float, default=0.75, help='GloVe model parameter')
parser.add_argument('--xmax', type=int, default=50, help='GloVe model parameter')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--numpy_random_seed', type=int, default=1234, help='Random Seed when minibatch < len(data)')
opt = parser.parse_args()
print(opt)

# Just as: http://pytorch.org/docs/master/notes/cuda.html
opt.cuda = not opt.disable_cuda and torch.cuda.is_available()
print(opt.cuda)

######## Part I : Data I/O ########
    
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
        
        if i%100 == 0: 
            print("{}/{} advance".format(i, len(listFiles)))
    
    print("Corpus Ready!!")    
    return(all_sentences)    


######## Part II : Coocurrance Matrix ########

def tokenize(string):
    string = string.lower()
    return string.split()

def extract_cooccurrences(dataset, word_map, word_to_index_map, amount_of_context=opt.context_window):
    print("Building cooccurrences matrix ...")    
    num_words = len(word_map)
    cooccurrences = np.zeros((num_words, num_words))
    nonzero_pairs = set()
    for example in dataset:
        words = tokenize(example)
        for target_index in range(len(words)):
            target_word = words[target_index]
            if target_word not in word_to_index_map:
                continue
            target_word_index = word_to_index_map[target_word]
            min_context_index = max(0, target_index - amount_of_context)
            max_word = min(len(words), target_index + amount_of_context + 1)
            for context_index in list(range(min_context_index, target_index)) + \
            list(range(target_index + 1, max_word)):
                context_word = words[context_index]
                if context_word not in word_to_index_map:
                    continue
                context_word_index = word_to_index_map[context_word]
                cooccurrences[target_word_index][context_word_index] += 1.0
                nonzero_pairs.add((target_word_index, context_word_index))
    return cooccurrences, list(nonzero_pairs)
                

######## Part III: Batchify data ########

def batch_iter(nonzero_pairs, cooccurrences, batch_size):
    start = -1 * batch_size
    dataset_size = len(nonzero_pairs)
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        word_i = []
        word_j = []
        counts = []
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
            random.shuffle(order)
        batch_indices = order[start:start + batch_size]
        batch = [nonzero_pairs[index] for index in batch_indices]
        for k in batch:
            counts.append(cooccurrences[k])
            word_i.append(k[0])
            word_j.append(k[1])
        yield [counts, word_i, word_j]
        
######## Part V: Model ######## 
        
class Glove(nn.Module):
    
    def __init__(self, embedding_dim, vocab_size, batch_size):
        
        super(Glove, self).__init__()    
        
        self.word_embeddings = None
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        
        self.wi = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.wj = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        self.bi = nn.Embedding(self.vocab_size, 1)
        self.bj = nn.Embedding(self.vocab_size, 1)
        
        self.init_weights()           
    
    def forward(self, new_wi, new_wj):      
        
        out_wi = torch.squeeze(self.wi(new_wi))
        out_wj = torch.squeeze(self.wj(new_wj))
        out_bi = torch.squeeze(self.bi(new_wi))
        out_bj = torch.squeeze(self.bj(new_wj))
        
        return out_wi, out_wj, out_bi, out_bj
        
        
    def init_weights(self):
        
        initrange = 0.1
        init_vars = [self.wi, self.wj, self.bi, self.bj]
        
        for var in init_vars:
            var.weight.data.uniform_(-initrange, initrange)
        

    def add_embeddings(self):
        
        self.word_embeddings = self.wi.weight + self.wj.weight
    
    def get_embeddings(self, index):
        if self.word_embeddings is None:
            self.add_embeddings()
        
        if opt.cuda:
            return self.word_embeddings.data[index, :].cpu().numpy()
        else:
            return self.word_embeddings.data[index, :].numpy()

    # Extra piece of code, to get whole matrix
    def get_matrix(self): 
        if self.word_embeddings is None:
            self.add_embeddings()
        
        if opt.cuda: 
            return self.word_embeddings.data.cpu().numpy()
        else:
            return self.word_embeddings.data.numpy()
        
######## Part VI: Training Loop ######## 

def training_loop(training_set, batch_size, num_epochs, model, optim, data_iter, xmax, alpha, optimizer):
    print("Training Model...")    
    step = 0
    epoch = 0
    losses = []
    total_batches = int(len(training_set) / batch_size)
    
    if opt.cuda:
        model = model.cuda()
        
    while epoch <= num_epochs:
        model.train()
        counts, words, co_words = next(data_iter)        
        words_var = Variable(torch.LongTensor(words))
        co_words_var = Variable(torch.LongTensor(co_words))
        
        if opt.cuda:
            words_var = words_var.cuda()
            co_words_var = co_words_var.cuda()
        
        model.zero_grad()

        wi, wj, bi, bj = model(words_var, co_words_var)
        counts_var = Variable(torch.FloatTensor([counts]))
        
        counts_fx = [1 if x >= xmax else (x/xmax)**alpha for x in counts]
        counts_fx_var = Variable(torch.FloatTensor([counts_fx]))
        
        if opt.cuda:
            counts_var = counts_var.cuda()
            counts_fx_var = counts_fx_var.cuda()
            
        loss = sum(torch.t(torch.mul((torch.mm(wi, torch.t(wj)).diag() + bi + bj - torch.log(counts_var))**2, counts_fx_var)))
                
        losses.append(loss.data[0])
        loss.backward()
        optimizer.step()
        
        if step % total_batches == 0:
            epoch += 1
            if epoch % 25 == 0:
                print( "Epoch:", (epoch),"/", (num_epochs) , "Avg Loss:", np.mean(losses)/(total_batches*epoch))
        
        if step % total_batches == 0:
            if epoch % 100 == 0:
                
                word_emebddings = model.get_matrix()
                pickle.dump(word_emebddings, open(opt.main_data_dir + "GloVe.p", "wb" ) )

        step += 1
        
######## Part VII: Train ######## 
        

def main():        
    train_examples = np.loadtxt(opt.main_data_dir + "/train.txt", dtype="str")
    if opt.minibatch < len(train_examples):
        np.random.seed(opt.numpy_random_seed)
        train_examples = np.random.choice(train_examples, opt.minibatch)
        
    corpus_sentences = loadCorpus(train_examples)    

    word_counter = collections.Counter()
    for example in corpus_sentences:
        word_counter.update(tokenize(example))
    vocabulary = [pair[0] for pair in word_counter.most_common()[0:opt.top_k]]
    index_to_word_map = dict(enumerate(vocabulary))
    word_to_index_map = dict([(index_to_word_map[index], index) for index in index_to_word_map])
    
    pickle.dump(word_to_index_map, open(opt.main_data_dir + "word_to_index_map.p", "wb" ) )      
    pickle.dump(index_to_word_map, open(opt.main_data_dir + "index_to_word_map.p", "wb" ) )   
    
    cooccurrences, nonzero_pairs = extract_cooccurrences(corpus_sentences, vocabulary, word_to_index_map)
    vocab_size = len(vocabulary)
        
    glove = Glove(opt.embedding_dim, vocab_size, opt.batchSize)
    glove.init_weights()
    optimizer = torch.optim.Adadelta(glove.parameters(), lr=opt.learning_rate)
    data_iter = batch_iter(nonzero_pairs, cooccurrences, opt.batchSize)
    
    training_loop(corpus_sentences, opt.batchSize, opt.num_epochs, glove, optimizer, data_iter, opt.xmax, opt.alpha, optimizer)
    
    word_emebddings = glove.get_matrix()
    
    pickle.dump(word_emebddings, open(opt.main_data_dir + "GloVe_final.p", "wb" ) ) 
    
if __name__ == "__main__" :
    main()
