# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 18:47:22 2020

@author: Shipra
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

with open('Amazon_Reviews.csv', 'r') as f:
    p = []
    for line in f:
            words = line.split(',')
            p.append((words[0]))
    print (p)

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(p)
print(matrix)
df1 = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names()) #dataframes using panda
print(df1)

df_array = df1.to_numpy()
color_map = plt.imshow(df_array) #weight matrix displayed.
color_map.set_cmap("jet")
plt.colorbar()
plt.show()   #visualization of matrix is displayed
#part a done count matrix

vec = CountVectorizer()
X = vec.fit_transform(p)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
df.to_csv('file1.csv')
print(df)

wordsGood = "great,like,good,love,fun"
wordsGood = wordsGood.split(",")
wordsBad = "awful,not,dont,only,bad"
wordsBad = wordsBad.split(",")
print('Good Words: ', wordsGood)
print('Bad Words: ',wordsBad)
words = wordsGood + wordsBad
print('Total Words:', words)
#part b good bad done

#part b tdift matrix
dfCount = df[words]
dftfidt = df1[words]
print(dfCount)
print(dftfidt)

dftfidtGBWordTemp = dftfidt[wordsGood]
dftfidtGBWordTempBad = dftfidt[wordsBad]
dftfidtGBWordSum = dftfidtGBWordTemp.sum(axis = 1)
dftfidtGBWordSumBad = dftfidtGBWordTempBad.sum(axis = 1) 
Sum = pd.concat([dftfidtGBWordSum,dftfidtGBWordSumBad], axis = 1)
print('Sum tfidf:',Sum)
X = Sum.values.tolist()

#

import copy
#
X = np.array(X)
#
n = X.shape[0]  #number of obejcts

k = 2
c = [[0.15,0.15],[0.3,0.3]]

c = np.array(c)
c = c.T

count = 0

def myKmeans(X,k,c):
    #features of data
    #fea = X_set.shape[1]
    # training data
    train = X.shape[0]
    #intializing an array with same shape as c
    c_old = np.zeros(c.shape)
    #new centers
    c_new = copy.deepcopy(c)
    #initializing the clusters to zero
    clusters = np.zeros(train)
    #distance of data set from cluster k
    distance = np.zeros((train,k))    
    #l2 norm between previous centers and updated centers is <=0.001 or number of iterations = 10000
    iterate = 0    
    while (np.linalg.norm(c_new-c_old) > 0.001):
        iterate = iterate+1
        print("Iteration: ",iterate)
        #check for the number of iterations
        if iterate == 10000:
            break

        for x in range(k):
            distance[:, x] = np.linalg.norm(X - c_new[x], axis = 1)
        #assigning the data sets to the closest center
        clusters = np.argmin(distance, axis = 1)
        print(clusters)
        #updating the old centers
        c_old = copy.deepcopy(c_new)
        #calculating the new centers by finding the average of data set for a cluster
        for x in range(k):
            c_new[x] = np.average(X[clusters == x], axis = 0 )
        

    #scatter plot
    #plt.scatter(X[:,0], X[:,1], c = 'k')   #, lable = 'Unclsuter Data'
    xaxis = []
    yaxis = []
    for a in X:
        for d in a:
            if d == a[0]:
                xaxis.append(d)
            if d == a[1]:
                yaxis.append(d)
    #print("xaxis :",xaxis)
    #print("yaxis :",yaxis)
    # color = [red, blue, green, yellow]
    
    for c,x,y in zip(clusters, xaxis, yaxis):
        if c == 0:
            plt.scatter(x, y, color = 'b')
            continue
        elif c == 1:
            plt.scatter(x, y, color = 'y')
            continue
        elif c == 2:
            plt.scatter(x, y, color = 'm')
            continue
        elif c == 3:
            plt.scatter(x, y, color = 'g')
            continue

    for x,y in c_new:
        plt.scatter(x, y, color = 'r')

    # plt.scatter(xplots, yplots)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()

clusters = myKmeans(X, k, c)
print('Number Iterations:', count)
