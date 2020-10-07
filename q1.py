
"""
Created on Sat Mar 28 2020
"""
import numpy as np
import matplotlib.pyplot as plt

dim = 0
h = [0.1, 1, 5, 10]
xi = []
for j in np.arange(-1, 10, 0.01):
    xi.append(j)

def mykde(X,h,x):
    sum_k = 0.0
    if (dim == 1):   # To check for 1-D or 2-D data
        N = len(X)   # number of data points in data set
        for i in range(N):
            ui = (x - X[i]) / h
            if (abs(ui) <= 0.5):        # condition to check value within hypercube(h=kernel functio represented as hypercube)
                k = 1
            else:
                k = 0
            sum_k = sum_k + k           # find summation i.e. total #of data points in cube
        px = float(sum_k / (N * h))     # estimated density at x
        return px, x
    
    if (dim == 2):
        N = len(X)
        for i in range(N):
            u1 = (x - X[i][0]) / h
            u2 = (x - X[i][1]) / h
            if (abs(u1) <= 0.5 and abs(u2) <= 0.5):
                k = 1
            else:
                k = 0
            sum_k = sum_k + k
        px = float(sum_k / (N * h * h))
        return px, x

# Part 1(a) : Generate N = 1000 Gaussian random data with u = 5 and sigma = 1. Test function mykde on data with h=[0.1,1,5,10].
u = 5
sigma = 1
N = 1000
dim = 1
X = np.random.normal(u, sigma, N)
p1 = [0.0] * (len(xi))
p2 = [0.0] * (len(xi))
p3 = [0.0] * (len(xi))
p4 = [0.0] * (len(xi))

for i in range(len(xi)):
    p1[i], xi[i] = mykde(X, h[0], xi[i])
    p2[i], xi[i] = mykde(X, h[1], xi[i])
    p3[i], xi[i] = mykde(X, h[2], xi[i])
    p4[i], xi[i] = mykde(X, h[3], xi[i])

figr1, axis = plt.subplots(nrows = 5, ncols = 1)
figr1.canvas.set_window_title('Part 1(A)')

axis[0].hist(X, 100, density = True, color = 'b')
axis[1].plot(xi, p1, color = 'b')
axis[2].plot(xi, p2, color = 'b')
axis[3].plot(xi, p3, color = 'b')
axis[4].plot(xi, p4, color = 'b')

axis[1].set_title('h=0.1 Part1(a)')
axis[2].set_title('h=1 Part1(a)')
axis[3].set_title('h=5 Part1(a)')
axis[4].set_title('h=10 Part1(a)')

plt.xlabel("x-axis data points(xi)")
plt.ylabel("y-axis estimated densities(px)")

# Part 1(b) : Generate N = 1000 1-D Gaussian random data with u1 = 5 and sigma1 = 1 and another Gaussian random data 
#with u2 = 0 and sigma2 = 0.2. Test function mykde on data with h=[0.1,1,5,10].
u1 = 5
sigma1 = 1
u2 = 0
sigma2 = 0.2
N1 = 500
N2 = 500
dim = 1
y1 = np.random.normal(u1, sigma1, N1)
y2 = np.random.normal(u2, sigma2, N2)
X = np.concatenate((y1, y2), axis = 0)
p1 = [0.0] * (len(xi))
p2 = [0.0] * (len(xi))
p3 = [0.0] * (len(xi))
p4 = [0.0] * (len(xi))

for i in range(len(xi)):
    p1[i], xi[i] = mykde(X, h[0], xi[i])
    p2[i], xi[i] = mykde(X, h[1], xi[i])
    p3[i], xi[i] = mykde(X, h[2], xi[i])
    p4[i], xi[i] = mykde(X, h[3], xi[i])

figr2, axis = plt.subplots(nrows = 5, ncols = 1)
figr2.canvas.set_window_title('Part 1(B)') 

axis[0].hist(X, 100, density = True, color = 'r')
axis[1].plot(xi, p1, color = 'r')
axis[2].plot(xi, p2, color = 'r')
axis[3].plot(xi, p3, color = 'r')
axis[4].plot(xi, p4, color = 'r')

axis[1].set_title('h=0.1 Part1(b)')
axis[2].set_title('h=1 Part1(b)')
axis[3].set_title('h=5 Part1(b)')
axis[4].set_title('h=10 Part1(b)')

plt.xlabel("x-axis data points(xi)")
plt.ylabel("y-axis estimated densities(px)")

# Part 2: 2 sets of 2-D Gaussian random data

u1 = [1,0]
u2 = [0,2.5]
sigma1 = [[0.9,0.4],[0.4,0.9]]
sigma2 = [[0.9,0.4],[0.4,0.9]]
N = 500
dim = 2
y1 = np.random.multivariate_normal(u1, sigma1, N)
y2 = np.random.multivariate_normal(u2, sigma2, N)
X = np.concatenate((y1,y2), axis = 0)
p1 = [0.0] * (len(xi))
p2 = [0.0] * (len(xi))
p3 = [0.0] * (len(xi))
p4 = [0.0] * (len(xi))

for i in range(len(xi)):
    p1[i], xi[i] = mykde(X, h[0], xi[i])
    p2[i], xi[i] = mykde(X, h[1], xi[i])
    p3[i], xi[i] = mykde(X, h[2], xi[i])
    p4[i], xi[i] = mykde(X, h[3], xi[i])

figr3, axis = plt.subplots(nrows = 5, ncols = 1)
figr3.canvas.set_window_title('Part 2') 

axis[0].hist(X, 100, density = True)
axis[1].plot(xi, p1, color = 'y')
axis[2].plot(xi, p2, color = 'y')
axis[3].plot(xi, p3, color = 'y')
axis[4].plot(xi, p4, color = 'y')

axis[1].set_title('h=0.1 Part 2')
axis[2].set_title('h=1 Part 2')
axis[3].set_title('h=5 Part 2')
axis[4].set_title('h=10 Part 2')

plt.xlabel("x-axis data points(xi)")
plt.ylabel("y-axis estimated densities(px)")
plt.show()

