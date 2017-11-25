# -*- coding: utf-8 -*-

import pandas as pd
import re
import numpy as np

"""
Created on Tue Nov 21 16:04:38 2017

@author: eduardofierro

Get target per document. 
"""

def readTable(main_data_dir):
    
    main_table = pd.read_csv(main_data_dir + "Main.csv")
    return main_table
    
def getTema(one_string):
    
    list_temas = re.split("[1-9*]\.-", one_string)
    list_temas = list_temas[1:len(list_temas)]
    
    return list_temas
    
def getTemaFromList(list_of_docs, main_table):
    
    list_int = [int(re.sub(".txt", "", x)) for x in list_of_docs]
    
    list_return = []
    for x in list_int:

        string = main_table.tema[x - 1]
        comisiones = getTema(string)
        list_return.append(comisiones)
        
    return list_return   
    
def genTargetSingle(list_strings, word): 
    
    current = 0
    for x in list_strings: 
        if word in x: 
            current = 1
    
    return current

def genTarget(list_strings, word): 
    
    return_list = []
    for x in list_strings: 
        value = genTargetSingle(x, word)
        return_list.append(value)
    
    return np.array(return_list)