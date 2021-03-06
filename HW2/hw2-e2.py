import numpy as np
import numpy.random as nr
import matplotlib.pyplot as pl

X = np.array([
	[1, 1], 
	[2, 2],
	[2, 0],
	[0, 1],
	[1, 0],
	[0, 0]
	])

Y = np.array([1, 1, 1, -1, -1, -1])

pl.rcParams['figure.figsize'] = (4.0, 4.0)
pl.figure()
axes = pl.gca()
axes.set_xlim([-0.5, 2.5])
axes.set_ylim([-0.5, 2.5])

for i in range(6):
	if Y[i] > 0:
		pl.plot(X[i, 0], X[i, 1], 'ro')
	else:
		pl.plot(X[i, 0], X[i, 1], 'bo')

x = np.linspace(-1,3)

def separator_function(x):
	return - x + 1.5
pl.plot(x, separator_function(x), 'k')

pl.show()