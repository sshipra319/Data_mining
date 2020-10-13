# -*- coding: utf-8 -*-
"""
Created on Sat May  9 00:08:36 2020

@author: Shipra Saini
"""

import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, alpha=0.001, iterations=100000):
        self.alpha = alpha
        self.iterations = iterations
        self.loss_threshold = 0.001
        self.iteration_count = 0
        self.fpr = []
        self.tpr = []
        self.total_cost=[]
        self.tot_iter = []
        self.norm_current_loss = []
        self.norm_grad =[]

    def activation_sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, predicted, actual):
        self.cost = (-actual * np.log(predicted) - (1 - actual) * np.log(1 - predicted)).mean()
        self.total_cost.append(self.cost)
        return self.cost

    def calc_scores(self, actual_y, predicted_y):
        predicted_y = np.round(predicted_y)
        predicted_y = predicted_y.tolist()
        actual_y = actual_y.tolist()

        fp = 0
        for i in range(500):
            if predicted_y[i] != actual_y[i]:
                fp += 1

        tp = 0
        for i in range(500):
            if predicted_y[i + 500] == actual_y[i + 500]:
                tp += 1

        tpr = tp / 500
        fpr = fp / 500

        self.fpr.append(fpr)
        self.tpr.append(tpr)

    def getScores(self):
        return self.fpr, self.tpr

    def learn(self, X, target):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        self.weights = np.ones(X.shape[1])

        previous_loss = float('inf')
        for self.iteration_count in range(self.iterations):
            net_val = np.dot(X, self.weights)
            prediction = self.activation_sigmoid(net_val)
            gradient = np.dot(X.T, (prediction - target)) / target.size
            self.weights -= self.alpha * gradient
            current_loss = self.loss(prediction, target)
            self.calc_scores(target, prediction)
            self.norm_grad.append(abs(gradient[0])+abs(gradient[1])+abs(gradient[2]))

            if (gradient == np.zeros(X.shape[1])).all():
                print("Entropy loss at epoch %s: %s" % (self.iteration_count + 1, self.loss(prediction, target)))
                print("Gradient is zero!")
                print("total no. of iterations run: ", self.iteration_count + 1)
                break

            if previous_loss - current_loss < self.loss_threshold:
                print("Entropy loss at epoch %s: %s" % (self.iteration_count + 1, self.loss(prediction, target)))
                print("Loss optimized is less than threshold!")
                print("total no. of iterations run: ", self.iteration_count + 1)
                break

            if self.iteration_count % 50 == 0:
                print("Entropy loss at epoch %s: %s" % (self.iteration_count + 1, self.loss(prediction, target)))

            if self.iteration_count == self.iterations - 1:
                print("Entropy loss at epoch %s: %s" % (self.iteration_count + 1, self.loss(prediction, target)))
                print("total no. of iterations run: ", self.iteration_count + 1)

            self.norm_current_loss.append(abs(current_loss-previous_loss))
            previous_loss = current_loss
            self.store_iter = self.iteration_count + 1
            self.tot_iter.append(self.store_iter)
        return self.norm_current_loss, self.tot_iter, self.norm_grad
            

    def predict(self, X):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return np.round(self.activation_sigmoid(np.dot(X, self.weights)))
    


def generate_data(mean, variance, count):
    return np.random.multivariate_normal(mean, variance, count)


def calculateAccuracy(predicted_y, test_y):
    predicted_y = predicted_y.tolist()
    test_y = test_y.tolist()

    count = 0
    for i in range(len(predicted_y)):
        if predicted_y[i] == test_y[i]:
            count += 1

    return (count / len(predicted_y)) * 100

max_iterations = 100000
train_x1 = generate_data([1, 0], [[1, 0.75], [0.75, 1]], 500)
train_x2 = generate_data([0, 1.5], [[1, 0.75], [0.75, 1]], 500)
train_X = np.vstack((train_x1, train_x2)).astype(np.float32)
train_y = np.hstack((np.zeros(500), np.ones(500)))

test_x1 = generate_data([1, 0], [[1, 0.75], [0.75, 1]], 250)
test_x2 = generate_data([0, 1.5], [[1, 0.75], [0.75, 1]], 250)
test_X = np.vstack((test_x1, test_x2)).astype(np.float32)
test_y = np.hstack((np.zeros(250), np.ones(250)))

print("\n\nLearning rate (Alpha): 1\nTotal Iterations: 100000")
LR = LogisticRegression(alpha=1, iterations=max_iterations)
LR.learn(train_X, train_y)
print("Final Weights: ", LR.weights)
predicted_y = LR.predict(test_X)
accuracy = calculateAccuracy(predicted_y, test_y)
print("Accuracy: ", accuracy)
f, ax = plt.subplots(1, figsize=(5, 5))
ax.set_title("Norm Gradient")
ax.plot(range(0, len(LR.norm_grad)), LR.norm_grad, 'g', label=r'$\alpha = 1$') 
ax.set_xlabel('Iterations')
ax.set_ylabel('Gradient')
ax.legend();
f, ax = plt.subplots(1, figsize=(5, 5))
plt.figure(figsize=(8, 8))
ax.set_title("Validation loss")
ax.plot(range(0, len(LR.total_cost)), LR.total_cost, 'r', label=r'$\alpha = 1$') 
ax.set_xlabel('Iterations')
ax.set_ylabel('Training Error')
ax.legend();
fpr, tpr = LR.getScores()
plt.figure(figsize=(5, 5))
plt.figure(figsize=(5, 5))
x_d = np.linspace(-3, 4, 50)
y_d = (LR.weights[0]*x_d)/LR.weights[1]
plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, alpha=.4)
plt.plot(x_d, y_d)
plt.show()

print("\n\nLearning rate (Alpha): 0.1\nTotal Iterations: 100000")
LR = LogisticRegression(alpha=0.1, iterations=max_iterations)
LR.learn(train_X, train_y)
print("Final Weights: ", LR.weights)
predicted_y = LR.predict(test_X)
accuracy = calculateAccuracy(predicted_y, test_y)
print("Accuracy: ", accuracy)
f, ax = plt.subplots(1, figsize=(5, 5))
ax.set_title("Norm Gradient")
ax.plot(range(0, len(LR.norm_grad)), LR.norm_grad, 'g', label=r'$\alpha = 0.1$') 
ax.set_xlabel('Iterations')
ax.set_ylabel('Gradient')
ax.legend();
f, ax = plt.subplots(1, figsize=(5, 5))
plt.figure(figsize=(8, 8))
ax.set_title("Validation loss")
ax.plot(range(0, len(LR.total_cost)), LR.total_cost, 'r', label=r'$\alpha = 0.1$') 
ax.set_xlabel('Iterations')
ax.set_ylabel('Training Error')
ax.legend();
fpr, tpr = LR.getScores()
plt.figure(figsize=(5, 5))
plt.figure(figsize=(5, 5))
x_d = np.linspace(-3, 4, 50)
y_d = (LR.weights[0]*x_d)/LR.weights[1]
plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, alpha=.4)
plt.plot(x_d, y_d)
plt.show()

print("\n\nLearning rate (Alpha): 0.01\nTotal Iterations: 100000")
LR = LogisticRegression(alpha=0.01, iterations=max_iterations)
LR.learn(train_X, train_y)
print("Final Weights: ", LR.weights)
predicted_y = LR.predict(test_X)
accuracy = calculateAccuracy(predicted_y, test_y)
print("Accuracy: ", accuracy)
f, ax = plt.subplots(1, figsize=(5, 5))
ax.set_title("Norm Gradient")
ax.plot(range(0, len(LR.norm_grad)), LR.norm_grad, 'g', label=r'$\alpha = 0.01$') 
ax.set_xlabel('Iterations')
ax.set_ylabel('Gradient')
ax.legend();
f, ax = plt.subplots(1, figsize=(5, 5))
plt.figure(figsize=(8, 8))
ax.set_title("Validation loss")
ax.plot(range(0, len(LR.total_cost)), LR.total_cost, 'r', label=r'$\alpha = 0.01$') 
ax.set_xlabel('Iterations')
ax.set_ylabel('Training Error')
ax.legend();
fpr, tpr = LR.getScores()
plt.figure(figsize=(5, 5))
plt.figure(figsize=(5, 5))
x_d = np.linspace(-3, 4, 50)
y_d = -(LR.weights[0]*x_d)/LR.weights[1]
plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, alpha=.4)
plt.plot(x_d, y_d)
plt.show()

print("\n\nLearning rate (Alpha): 0.001\nTotal Iterations: 100000")
LR = LogisticRegression(alpha=0.001, iterations=max_iterations)
LR.learn(train_X, train_y)
print("Final Weights: ", LR.weights)
predicted_y = LR.predict(test_X)
accuracy = calculateAccuracy(predicted_y, test_y)
print("Accuracy: ", accuracy)
f, ax = plt.subplots(1, figsize=(5, 5))
ax.set_title("Norm Gradient")
ax.plot(range(0, len(LR.norm_grad)), LR.norm_grad, 'g', label=r'$\alpha = 0.001$') 
ax.set_xlabel('Iterations')
ax.set_ylabel('Gradient')
ax.legend();
f, ax = plt.subplots(1, figsize=(5, 5))
plt.figure(figsize=(8, 8))
ax.set_title("Validation loss")
ax.plot(range(0, len(LR.total_cost)), LR.total_cost, 'r', label=r'$\alpha = 0.001$') 
ax.set_xlabel('Iterations')
ax.set_ylabel('Training Error')
ax.legend();
fpr, tpr = LR.getScores()
plt.figure(figsize=(5, 5))
plt.figure(figsize=(5, 5))
x_d = np.linspace(-3, 4, 50)
y_d = (LR.weights[0]*x_d)/LR.weights[1]
plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, alpha=.4)
plt.plot(x_d, y_d)
plt.show()

plt.figure(figsize=(5, 5))
plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, alpha=.4)
plt.show()

