### The output of your program should be written in a file as follows.
#   for question 'i', write the output in 'problem-i.txt' file (e.g., 'problem-1a.txt')
fo = open('problem-i.txt', 'w')

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