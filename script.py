import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import pickle
import os
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import time

eta = 0.2
count = 0
t = np.zeros((50000,716))
speedup = False

def preprocess():
    """
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """
    #path = "C:\\Users\\admin\\PycharmProjects\\PA3\\"
    path = os.getcwd()
    mat = loadmat(os.path.join(path,'mnist_all.mat'))  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 * (1.0 + np.exp(-z))**-1

# @profile
def blrObjFunction(W, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    if(len(args)==2):
        train_data, labeli = args
        train_data = np.hstack((np.ones((train_data.shape[0], 1)), train_data))
    if(len(args)==3):
        train_data, labeli, pre_set = args


    n_data = train_data.shape[0]

    W = np.reshape(W, (train_data.shape[1],1))
    P1 = sigmoid(np.dot(train_data, W))
    P2 = 1.0 - P1

    P1Likelihood = np.multiply(np.log(P1),labeli)
    P2Likelihood = np.multiply(np.log(P2),(1.0-labeli))

    P1 -= labeli
    P1 = np.expand_dims(P1, 1)
    error = np.sum(P1Likelihood + P2Likelihood) / -n_data
    error_grad = np.multiply(np.dot(P1.T, train_data) , n_data**-1)
    error_grad = error_grad.flatten()

    global count
    count += 1
    return error, error_grad


def blrPredict(W, data, pre_set = False):
    """
     blrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix


    """

    global count
    n_data = data.shape[0]
    n_feature = data.shape[1]
    label = np.zeros((data.shape[0], 1))
    if not pre_set:
        data = np.hstack((np.ones((data.shape[0], 1)), data))

    P1 = sigmoid(np.dot(data, W))
    P1Likelihood = np.log(P1)
    P1Likelihood -= np.ones_like(P1Likelihood)

    count +=1
    label = np.argmax(P1Likelihood, 1)
    label = np.expand_dims(label,1)

    return label


def mlrObjFunction(W, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    if(len(args)==2):
        train_data, labeli = args
        train_data = np.hstack((np.ones((train_data.shape[0], 1)), train_data))
    if(len(args)==3):
        train_data, labeli, pre_set = args

    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    W = W.reshape((n_feature, 10))
    mul = np.dot(train_data, W)
    P1 = np.exp(mul)
    P1Sum = np.expand_dims(np.sum(P1, 1),1)
    P1 = P1 * P1Sum**-1
    P1Likelihood = P1
    error = - (np.sum(labeli * P1Likelihood) * n_data**-1)
    error_grad = np.dot(train_data.T, (P1Likelihood - labeli)) * n_data**-1
    error_grad = error_grad.flatten()

    global count
    count += 1
    return error, error_grad


def mlrPredict(initialWeights, train_data, pre_set=False):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    n_data = train_data.shape[0]
    label = np.ones((n_data, 10))
    if not pre_set:
        train_data = np.hstack((np.ones((n_data, 1)), train_data))
    P1 = np.zeros((train_data.shape[0], 10))


    P1 = np.exp(np.dot(train_data,initialWeights))
    P1Sum = np.sum(P1, 1)
    P1Sum = np.expand_dims(np.sum(P1, 1),1)
    P1 = P1/P1Sum

    label = np.argmax(P1, 1)
    label = np.expand_dims(label,1)
    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
print("Preprocess Complete...")
# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}

if speedup:
    train_data = np.hstack((np.ones((n_train, 1)), train_data))
    validation_data = np.hstack((np.ones((validation_data.shape[0], 1)), validation_data))
    test_data = np.hstack((np.ones((test_data.shape[0], 1)), test_data))

t0 = time.time()

for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    if speedup:
        args = (train_data, labeli, True)
    count = 0
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    print("Executed for class : " + str(i)+" Iterations : "+ str(count))
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

t1 = time.time()
print("BLR Learn completed in :"+str(t1-t0))
print("Writing to pickle file [blr]: init")
save_path = os.path.join(os.getcwd(), "params.pickle")
save_data = {'W': W,}
with open(save_path, 'wb') as handle:
    pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Writing to pickle file : complete")


data2 = {}
save_path = os.path.join(os.getcwd(), "params.pickle")
print("Reading from pickle file : init")
with open(save_path, 'rb') as handle:
    data2 = pickle.load(handle)
W = data2['W']
print("Reading from pickle file : complete")
#

t0 = time.time()
count = 0
# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data, speedup)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data, speedup)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data, speedup)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
t1 = time.time()
print("BLR Predict completed in :"+str(t1-t0))

"""
Script for Support Vector Machine
"""


print('\n\n--------------SVM-------------------\n\n')

train_label = train_label.ravel()
validation_label = validation_label.ravel()
test_label = test_label.ravel()

t0 = time.time()
# ------- Linear -----------
classifier = SVC(kernel='linear')
classifier.fit(train_data, train_label)

print('\n Linear Kernel : \n------------')
print('\n Training set Accuracy:' + str(100 * classifier.score(train_data, train_label)) + '%')
# Find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100 * classifier.score(validation_data, validation_label)) + '%')
# Find the accuracy on Testing Dataset
print('\n Testing set Accuracy:' + str(100 * classifier.score(test_data, test_label)) + '%')
t1 = time.time()

print("SVC Linear completed in :"+str(t1-t0))

# ---------- RBF gamma = 1 ----------
t0 = time.time()
classifier = SVC(kernel='rbf', gamma=1.0)
classifier.fit(train_data, train_label)

print('\n RBF Kernel : Gamma = 1\n------------')
# Find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100 * classifier.score(train_data, train_label)) + '%')
# Find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100 * classifier.score(validation_data, validation_label)) + '%')
# Find the accuracy on Testing Dataset
print('\n Testing set Accuracy:' + str(100 * classifier.score(test_data, test_label)) + '%')
t1 = time.time()
print("SVC RBF Gamma = 1 completed in :"+str(t1-t0))

# ---------- RBF gamma = auto ----------
t0 = time.time()
classifier = SVC(kernel='rbf')
classifier.fit(train_data, train_label)

print('\n RBF Kernel : Gamma = auto\n------------')
# Find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100 * classifier.score(train_data, train_label)) + '%')
# Find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100 * classifier.score(validation_data, validation_label)) + '%')
# Find the accuracy on Testing Dataset
print('\n Testing set Accuracy:' + str(100 * classifier.score(test_data, test_label)) + '%')
t1 = time.time()

print("SVC RBF Gamma = auto completed in :"+str(t1-t0))


accuracies = np.zeros((11, 3))
C_vals = np.array([1.0,10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
t0 = time.time()

for index, i in enumerate(C_vals):
    # ---------- RBF gamma = auto, c = varies ----------
    print('\n BRF Kernel : C : '+str(i)+'\n------------')
    classifier = SVC(kernel='rbf', C=i)
    classifier.fit(train_data, train_label)

    print('Training complete')
    # Find the accuracy on Training Dataset
    accuracies[index][0] = 100 * classifier.score(train_data, train_label)
    print('\n Training set Accuracy:' + str(accuracies[index][0]) + '%')
    # Find the accuracy on Validation Dataset
    accuracies[index][1] = 100 * classifier.score(validation_data, validation_label)
    print('\n Validation set Accuracy:' + str(accuracies[index][1]) + '%')
    # Find the accuracy on Testing Dataset
    accuracies[index][2] = 100 * classifier.score(test_data, test_label)
    print('\n Testing set Accuracy:' + str(accuracies[index][2]) + '%')

t1 = time.time()
print("SVC RBF C range completed in :"+str(t1-t0))

print("Writing to pickle file [SVC]: init")
save_path = os.path.join(os.getcwd(), "params_SVC.pickle")
save_data = {'acc': accuracies,}
with open(save_path, 'wb') as handle:
    pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Writing to pickle file : complete")

data2 = {}
save_path = os.path.join(os.getcwd(), "params_SVC.pickle")
print("Reading from pickle file : init")
with open(save_path, 'rb') as handle:
    data2 = pickle.load(handle)
accuracies = data2['acc']
print("Reading from pickle file : complete")


print(accuracies)
plt.plot(C_vals, accuracies)
plt.legend(['training_data','validation_data','test_data'])
# plt.show()

plt.savefig('SVC.png')

print("Writing to pickle file [SVC]: init")
save_path = os.path.join(os.getcwd(), "params_SVC.pickle")
save_data = {'acc': accuracies,}
with open(save_path, 'wb') as handle:
    pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Writing to pickle file : complete")

# -----------------------------------------------------------#


'''
Script for Extra Credit Part

'''
# FOR EXTRA CREDIT ONLY

train_label = np.expand_dims(train_label,1)
validation_label = np.expand_dims(validation_label,1)
test_label = np.expand_dims(test_label,1)

W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}
args_b = (train_data, Y)
if speedup:
    args_b = (train_data, Y, True)
count = 0
t0 = time.time()
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
print("Executed MLR :  Iterations : "+ str(count))
W_b = nn_params.x.reshape((n_feature + 1, n_class))
t1 = time.time()
print("MLR Learn completed in :"+str(t1-t0))
print("Writing to pickle file [mlr]: init")
save_path = os.path.join(os.getcwd(), "params_bonus.pickle")
save_data = {'W': W_b,}
with open(save_path, 'wb') as handle:
    pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Writing to pickle file : complete")

data2 = {}
print("Reading from pickle file : init")
with open(save_path, 'rb') as handle:
    data2 = pickle.load(handle)
W_b = data2['W']
print("Reading from pickle file : complete")


t0 = time.time()
# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data, speedup)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data, speedup)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data, speedup)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
t1 = time.time()
print("MLR Predict completed in :"+str(t1-t0))
