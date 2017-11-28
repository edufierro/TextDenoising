# -*- coding: utf-8 -*-

import pickle
import numpy as np
import warnings
from sklearn.preprocessing import normalize

'''

This code replicates Algorithm 2 from "Robust Matrix Completition and Corrupted Columns"

http://www.icml-2011.org/papers/469_icmlpaper.pdf 

'''

def getObserved(mat): 
    
    observed = []
    observed_complement = []
    for x in range(0, mat.shape[0]):
        for y in range(0, mat.shape[1]):
            
            if mat[x,y] == 0: 
                observed.append((x,y)) # Touple with index_row; index_col
            else: 
                observed_complement.append((x,y))
    
    return observed, observed_complement 

def POfobs(mat, list_obs): 
    
    zeros = np.zeros(mat.shape)
    for x in list_obs:
        zeros[x] = mat[x]
                
    return zeros

def matrix_pq_norm(mat, p=1, q=2):
    
    value = np.sum(np.sum(np.abs(mat)**p, axis=1)**(q/p))**(1/q)
    
    return value

def colTreshold(mat, epsilon):
    
    mat_copy = mat.copy()
    for x in range(0, mat.shape[1]):
        
        col = mat[:,x]
        col_norm = np.linalg.norm(col)
        if col_norm <= epsilon:
            col = np.zeros(col.shape[0])
        else:
            col = col - (epsilon * col / col_norm)
            
        mat_copy[:,x] = col
    
    return mat_copy

def valueTreshold(mat, epsilon):
    
    mat_copy = mat.copy()
    
    for x in range(0, mat.shape[0]):
        for y in range(0, mat.shape[1]):
            value = mat[x,y]
        
            if np.abs(value) <= epsilon: 
                sub = 0
            else: 
                sub = value - (epsilon*value/np.abs(value))
            
            mat_copy[x,y] = sub
            
    return mat_copy

def convergence_criterion(M, Ek, Lk, Ck):
    
    norm1 = np.linalg.norm(M - Ek - Lk - Ck )
    norm2 = np.linalg.norm(M)
    condition = (norm1 / norm2) <= 10e-6
    
    return condition

def ALM_RoMaCo(M, lambda_fun = 0.5, alpha=1.1, tol=50):
    
    # First alpha = 1.1 is the same as paper. 
    # Lambda is not specified, but assumed 0<lambda<1

    # Initialize #
    Yk = np.zeros(M.shape)
    Lk = np.zeros(M.shape)
    Ck = np.zeros(M.shape)
    Ek = np.zeros(M.shape)
    k = 0
    convergence = False
    converged = True

    # Initialize U --> Same as paper, but with normalized columns
    uk = matrix_pq_norm(normalize(M), p = 1, q = 2)**(-1)
    obs_indexes, obs_indexes_complement = getObserved(M)
    
    while convergence is False:
        
        U,s,V = np.linalg.svd(M - Ek - Ck + (1/uk)*Yk)
        S = np.zeros((U.shape[0], V.shape[0]), dtype=float)
        minS = np.min((V.shape[0], U.shape[0]))
        S[:minS, :minS] = np.diag(s)
        
        Lk = np.dot(U, np.dot(valueTreshold(S,1/uk), V))
        Ck = colTreshold(M - Ek - Lk + Yk/uk, lambda_fun / uk)
        Ek = POfobs(M - Lk - Ck - (1/uk)*Yk, obs_indexes_complement)
        Yk = Yk + uk*(M - Ek - Lk - Ck)
        uk = alpha * uk
        k += 1
        print("Iteration = {}".format(k))
        convergence = convergence_criterion(M, Ek, Lk, Ck)
        
        if tol == k: 
            convergence = True # Force convergence
            print("Model didn't converged")
            converged = False
    
    if converged == True:
        print("Model Converged!")
    else: 
        print("WARNING: Model stopped after {} iterations. Did not converged".format(k))
        
    return Lk, Ck