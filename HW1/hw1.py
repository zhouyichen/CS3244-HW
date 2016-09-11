import numpy as np
import numpy.random as nr
import matplotlib.pyplot as pl
# %matplotlib inline
# Plotting with style! 
import seaborn as sb 

# Size the plot appropriately for online display
pl.rcParams['figure.figsize'] = (12.0, 10.0)

nr.seed(3244)

#    Write your solution to the programming assignment here.  We've suggested some cells that you can add 
#    to your notebook as single line comments below.
#    Please place all of your cells to be run in a linear, unintervened order, such that we can automate
#    the running and grading of the assignment.

# load datasets code
from io import StringIO

def read_data_file(file_name):
	f = open(file_name, "r")
	mat = np.loadtxt(StringIO(f.read()))
	X = mat[:, :-1]
	X = np.insert(X, 0, 1, axis = 1)
	y = np.matrix(mat[:, -1]).T
	return (X, y)

X_train, Y_train = read_data_file("hw1-train.dat")
N, M = X_train.shape

X_test, Y_test = read_data_file("hw1-test.dat")

# LR code

def lr(x, y, w, eta):
	'''
	    Input: 
	        xn : Data points
	        yn : Classification of the previous data points
	        w  : initial weight
	        eta : step size
	    Output: 
	        w : updated weight
	'''
	n = x.shape[0]
	g = - np.sum(
			np.multiply(x, y) / (1 + np.exp(np.multiply(y, (np.dot(x, w))))),
			0
		) / n
	w = w - eta * g.T
	return w
    
# Evaluation code

# calculate the in sample error given w
def calculate_error(xn, yn, w):
	N = xn.shape[0]
	e = np.sum(np.log(1 + np.exp(np.multiply(-yn, (np.dot(xn, w)))))) / N
	# print(e)
	return e

# sigmoid function which can be applied to a vector
def sigmoid(s):
	exp_s = np.exp(s)
	return np.divide(exp_s, (1 + exp_s))

# calculates the E_out given data, classification, and weights
def out_error(xn, yn, w):
	N = xn.shape[0]
	prediction = (sigmoid(np.dot(xn, w)) > 0.5)
	real_class = (yn > 0)
	e = np.sum(prediction != real_class) / N
	return e

def evaluate_batch_GD(eta, iteration):
	'''
		Run batch GD using X_train and Y_train
		Input:
			eta: step size
			iteration: number of iteration of SGD
		output:
			w : final weights
			E_out: Error for out samples using X_test and Y_test
	'''
	w = np.zeros([M, 1])
	for i in range(iteration):
	    # calculate_error(X_train, Y_train, w)
	    w = lr(X_train, Y_train, w, eta)
	E_out = out_error(X_test, Y_test, w)
	return (w.T.tolist()[0], E_out)

def evaluate_deterministic(eta, iteration):
	'''
		Run deterministic gradient descent using X_train and Y_train
		Input:
			eta: step size
			iteration: number of iteration of gradient descent
		output:
			w : final weights
			E_out: Error for out samples using X_test and Y_test
	'''
	w = np.zeros([M, 1])
	for i in range(iteration):
		# calculate_error(X_train, Y_train, w)
		index = i % N
		w = lr(np.mat(X_train[index, :]), Y_train[index], w, eta)
	E_out = out_error(X_test, Y_test, w)
	return (w.T.tolist()[0], E_out)


w, E_out = evaluate_batch_GD(0.05, 2333)
print("a: eta = 0.05, T = 2333")
print("Eventual weight vector:")
print(w)
print("E_out:", E_out, "\n")

w, E_out = evaluate_batch_GD(0.005, 2333)
print("b: eta = 0.005, T = 2333")
print("Eventual weight vector:")
print(w)
print("E_out:", E_out, "\n")

w, E_out = evaluate_deterministic(0.05, 2333)
print("c1: eta = 0.05, T = 2333, deterministic")
print("Eventual weight vector:")
print(w)
print("E_out:", E_out, "\n")

w, E_out = evaluate_deterministic(0.005, 2333)
print("c2: eta = 0.05, T = 2333, deterministic")
print("Eventual weight vector:")
print(w)
print("E_out:", E_out, "\n")

# Plotting (if any) code
# obtain the separator function from given weight
def separator_function(w):
	def f(x):
		b = - w[0, 0] / w[2, 0]
		a = - w[1, 0] / w[2, 0]
		return a * x + b
	return f

def part_d():
	x = np.linspace(0,1)
	pl.figure()
	axes = pl.gca()
	axes.set_xlim([-0.2, 1.2])
	axes.set_ylim([-0.2, 1.2])

	# draw the points
	for i in range(N):
		if Y_train[i] > 0:
			pl.plot(X_train[i, 1], X_train[i, 2], 'ro')
		else:
			pl.plot(X_train[i, 1], X_train[i, 2], 'bo')

	# draw the separator at each 2333 step
	w = np.zeros([M, 1])
	for i in range(1, 2333 + 1):
	    w = lr(X_train, Y_train, w, 0.05)
	    if not i % 333:
	    	color = str(0.8 - (i / 333 - 1) * (0.8 / 7))
	    	print(color)
	    	pl.plot(x, separator_function(w)(x), color)

	# draw the separator after the last iteration
	pl.plot(x, separator_function(w)(x), 'k')
	pl.title('Part(d)')
	pl.show()

part_d()
