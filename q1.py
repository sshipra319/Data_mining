# -*- coding: utf-8 -*-
"""
Created on Wed Feb  28 18:29:50 2020

@author: Shipra
"""
import numpy as np
import copy
import matplotlib.pyplot as plt

N = 500
u1 = [1,0]
u2 = [0,1.5]
cov1 = [[0.9,0.4],[0.4,0.9]]
cov2 = [[0.9,0.4],[0.4,0.9]]

# 2-D guassian random data
x = np.random.multivariate_normal(u1, cov1, N)
y = np.random.multivariate_normal(u2, cov2, N)

X_set = np.concatenate((x,y), axis = 0)

K_value = input("Enter the number of clusters: ")
k = int(K_value)
c = []

for x in range(k):
    centers_val = input("Enter the centers for cluster "+ str(x+1) +" : ")
    centers = []
    for x in centers_val.split(" "):
        centers.append(float(x))
    c.append(centers)
c = np.asarray(c, dtype = np.float64)

def myKmeans(X,k,c):
    #features of data
    #fea = X_set.shape[1]
    # training data
    train = X_set.shape[0]
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
            distance[:, x] = np.linalg.norm(X_set - c_new[x], axis = 1)
        #assigning the data sets to the closest center
        clusters = np.argmin(distance, axis = 1)
        print(clusters)
        #updating the old centers
        c_old = copy.deepcopy(c_new)
        #calculating the new centers by finding the average of data set for a cluster
        for x in range(k):
            c_new[x] = np.average(X_set[clusters == x], axis = 0 )
        
    print("Centers after kmeans: ", c_new)

    #scatter plot
    xaxis = []
    yaxis = []
    for a in X_set:
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


myKmeans(X_set,k,c)