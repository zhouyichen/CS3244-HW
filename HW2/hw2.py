import numpy as np
from io import StringIO

def read_q1_file(file_name):
	f = open(file_name, "r")
	mat = np.loadtxt(StringIO(f.read()))
	X = mat[:, 1:]
	# X = np.insert(X, 0, 1, axis = 1)
	y = np.matrix(mat[:, 0]).T
	return (X, y)

def run_svm_and_write_output(q_number, dataTrain, labelTrain, dataTest, labelTest, cost, kernel, gamma, degree):
	### The output of your program should be written in a file as follows.
	#   for question 'i', write the output in 'problem-i.txt' file (e.g., 'problem-1a.txt')
	fo = open('problem-' + q_number + '.txt', 'w')

	# train your svm
	# (n.b., svmTrain, svmPredict are not previously defined;
	# you will have to supply code to implement them)
	svmModel, totalSV  = svmTrain(dataTrain, labelTrain, cost, kernel, gamma, degree)

	# test on the training data
	trainAccuracy = svmPredict(dataTrain, labelTrain, svmModel)

	# test on your test data
	testAccuracy = svmPredict(dataTest, labelTest, svmModel)

	# report your results in the file
	fo.write("Kernel: "+ str(kernel)+"\n")
	fo.write("Cost: "+ str(cost)+ "\n")
	fo.write("Number of Support Vectors: "+ str(totalSV)+"\n")
	fo.write("Train Accuracy: "+ str(trainAccuracy)+"\n")
	fo.write("Test Accuracy: " + str(testAccuracy)+"\n")

q1_train_data, q1_train_lable = read_q1_file('hw2-1-train.txt')
q1_test_data, q1_test_lable = read_q1_file('hw2-1-test.txt')