#!/usr/bin/env python

import numpy as np
import csv


def array(infile):
    result = []
    with open(infile) as csvfile:
        reader = csv.reader(csvfile) # change contents to floats
        for row in reader: # each row is a list
            result.append(row)
    return(result)


def rewrite(infile):
    edit  = []
    for i in range(len(infile)):
        string = str(infile[i])
        row = string.replace('[','')
        row = row.replace(']','')
        row = row.replace('\"','')
        row = row.replace('\n','')
        row = row.replace(' ','')
        row = row.replace('\"','')
        row = row.replace('?','')
        row = row.replace('\'', '')
        edit.append(row)
    return(edit)


def esp_entier(array):
    s=0
    entier = [array[0][11],0]
    array[0].append(s)
    for i in range(1,len(array)-2):
        array[i].append(s)
        if array[i][11] != array[i+1][11]:
            s+=1
            entier.append([array[i+1][11],s])
    array[len(array)-1].append(s)
    array[len(array)-2].append(s)
    return(entier,array)

def labels(array):
    label = []
    for i in range(len(array)-1):
        label.append([array[i][12],array[i][13],array[i][15]])
    return(label)




