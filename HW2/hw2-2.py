import numpy as np
from io import StringIO
from sklearn import svm
import csv

DIM = 5000

def read_q2_file(file_name, rows):
	X = np.zeros([rows, DIM])
	Y = np.zeros(rows)
	i = 0
	with open(file_name) as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		for row in reader:
			Y[i] = float(row[0])
			for elem in row[1: -1]:
				a, b = elem.split(':')
				X[i, int(a) - 1] = float(b)
			i += 1
	return (X, Y)
		

def svmTrain(dataTrain, labelTrain, cost, kernel, gamma, degree):
	clf = svm.SVC(C=cost, kernel=kernel, gamma=gamma, degree=degree, coef0=1.0)
	svc = clf.fit(dataTrain, labelTrain)
	totalSV = svc.support_.size
	return (clf, totalSV)

def svmPredict(data, lable, svmModel):
	predictions = svmModel.predict(data)
	accuracy = np.sum(predictions == lable) / lable.size
	return accuracy

def print_result(totalSV, trainAccuracy, testAccuracy):
	print('Number of SV =', totalSV, ', trainAccuracy =', trainAccuracy, ', testAccuracy =', testAccuracy,
		', E_in =', 1 - trainAccuracy, ', E_out =', 1 - testAccuracy)

def run_svm_and_write_output(q_number, dataTrain, labelTrain, dataTest, labelTest, cost, kernel, gamma, degree, write=True):
	# train your svm
	# (n.b., svmTrain, svmPredict are not previously defined;
	# you will have to supply code to implement them)
	svmModel, totalSV  = svmTrain(dataTrain, labelTrain, cost, kernel, gamma, degree)

	# test on the training data
	trainAccuracy = svmPredict(dataTrain, labelTrain, svmModel)

	# test on your test data
	testAccuracy = svmPredict(dataTest, labelTest, svmModel)

	if write:
		### The output of your program should be written in a file as follows.
		#   for question 'i', write the output in 'problem-i.txt' file (e.g., 'problem-1a.txt')
		fo = open('problem-' + q_number + '.txt', 'w')
		# report your results in the file
		fo.write("Kernel: "+ str(kernel)+"\n")
		fo.write("Cost: "+ str(cost)+ "\n")
		fo.write("Number of Support Vectors: "+ str(totalSV)+"\n")
		fo.write("Train Accuracy: "+ str(trainAccuracy)+"\n")
		fo.write("Test Accuracy: " + str(testAccuracy)+"\n")
	print_result(totalSV, trainAccuracy, testAccuracy)

q2_train_data, q2_train_lable = read_q2_file('hw2-2-train.txt', 6000)
q2_test_data, q2_test_lable = read_q2_file('hw2-2-test.txt', 1000)

print('Problem 2d')
print('Linear Kernel:')
run_svm_and_write_output('2d', q2_train_data, q2_train_lable, q2_test_data, q2_test_lable, 1, 'linear', 'auto', 1)

print('\nProblem 2e')
print('RBF Kernel:')
run_svm_and_write_output('2e1', q2_train_data, q2_train_lable, q2_test_data, q2_test_lable, 1, 'rbf', 0.001, 1)
print('\nPolynomial Kernel:')
run_svm_and_write_output('2e2', q2_train_data, q2_train_lable, q2_test_data, q2_test_lable, 1, 'poly', 1.0, 2)
