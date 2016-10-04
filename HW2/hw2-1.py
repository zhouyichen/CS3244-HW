import numpy as np
from io import StringIO
from sklearn import svm

def read_q1_file(file_name):
	f = open(file_name, "r")
	mat = np.loadtxt(StringIO(f.read()))
	X = mat[:, 1:]
	# X = np.insert(X, 0, 1, axis = 1)
	y = mat[:, 0]
	return (X, y)

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

q1_train_data, q1_train_lable = read_q1_file('hw2-1-train.txt')
q1_test_data, q1_test_lable = read_q1_file('hw2-1-test.txt')


#Q1a
print('Problem 1a')
print('All training data and lables:')
run_svm_and_write_output('1a', q1_train_data, q1_train_lable, q1_test_data, q1_test_lable, 1, 'linear', 'auto', 1)
print()

subset_sizes = [50, 100, 200, 800]
for s in subset_sizes:
	print('Subset size =', s, ':')
	sub_train_data = q1_train_data[:s]
	sub_train_lable = q1_train_lable[:s]
	run_svm_and_write_output('1a_'+str(s), sub_train_data, sub_train_lable, q1_test_data, q1_test_lable, 1, 'linear', 'auto', 1, False)
	print()

print('\nProblem 1b')
C_list = [0.0001, 0.001, 0.01, 1.0]
for C in C_list:
	print('C =', C)
	print('Q = 2:')
	run_svm_and_write_output('1b_2_'+str(C), q1_train_data, q1_train_lable, q1_test_data, q1_test_lable, C, 'poly', 1.0, 2, False)
	print('Q = 5:')
	run_svm_and_write_output('1b_5_'+str(C), q1_train_data, q1_train_lable, q1_test_data, q1_test_lable, C, 'poly', 1.0, 5, False)
	print()
	
print('\nProblem 1c')
C_list = [0.01, 1, 100, 10000, 1000000]
for C in C_list:
	print('C =', C)
	run_svm_and_write_output('1c_'+str(C), q1_train_data, q1_train_lable, q1_test_data, q1_test_lable, C, 'rbf', 1.0, 1, False)
	print()

