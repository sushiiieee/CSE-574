import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

import time
import os
import sys
import pickle
import math

count = 0
prev_reg = 0
gl_n_hidden = 16
gl_lambdaval = 0.2

if len(sys.argv) > 2 :
    gl_n_hidden = int(sys.argv[1])
    gl_lambdaval = float(sys.argv[2])

print("Running with params : ", gl_n_hidden, " " ,gl_lambdaval)


def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return np.reciprocal((1+np.exp(-1*z) ))
    
    

def preprocess():
    """ Input:
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

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary


    '''
        Feature Selection : Remove the two end columns
    '''
    rem = []
    for i in range(28):
        rem.append(i*28)
        rem.append((i+1)*28-1)

    test = []
    train = []
    true_label = np.eye(10)
    test_data = np.zeros((1,28*28 - len(rem)))
    test_label = np.zeros((1,10))
    train_data = np.zeros((1,28*28 - len(rem)))
    train_label = np.zeros((1,10))
    validation_data = np.zeros((1,28*28 - len(rem)))
    validation_label = np.zeros((1,10))

    test.append(mat.get('test0'))
    test.append(mat.get('test1'))
    test.append(mat.get('test2'))
    test.append(mat.get('test3'))
    test.append(mat.get('test4'))
    test.append(mat.get('test5'))
    test.append(mat.get('test6'))
    test.append(mat.get('test7'))
    test.append(mat.get('test8'))
    test.append(mat.get('test9'))


    train.append(mat.get('train0'))
    train.append(mat.get('train1'))
    train.append(mat.get('train2'))
    train.append(mat.get('train3'))
    train.append(mat.get('train4'))
    train.append(mat.get('train5'))
    train.append(mat.get('train6'))
    train.append(mat.get('train7'))
    train.append(mat.get('train8'))
    train.append(mat.get('train9'))


    for i in range(0,10):
        test[i] = np.delete(test[i],rem,1)
        train[i] = np.delete(train[i],rem,1)

        train_range = range(train[i].shape[0])
        aperm = np.random.permutation(train_range)
        validation_size = train[i].shape[0]//6

        validation_data = np.append(validation_data, train[i][aperm[0:validation_size],:],0)
        validation_label = np.append(validation_label, np.full((validation_size,10), true_label[i]),0)

        train_data = np.append(train_data, train[i][aperm[validation_size:],:],0)
        train_label = np.append(train_label, np.full((train[i].shape[0]-validation_size,10), true_label[i]),0)

        test_data = np.append(test_data, test[i],0)
        test_label = np.append(test_label, np.full((test[i].shape[0],10), true_label[i]),0)

    train_data = train_data[1:,:]/255
    train_label = train_label[1:,:]
    train_label = np.expand_dims(train_label, 1)

    validation_data = validation_data[1:,:]/255
    validation_label = validation_label[1:,:]
    validation_label = np.expand_dims(validation_label, 1)

    test_data = test_data[1:,:]/255
    test_label = test_label[1:,:]
    test_label = np.expand_dims(test_label, 1)

    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    global count
    global prev_reg
    print("Round : ",count)
    count += 1
    start_time = int(round(time.time() * 1000))
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    obj_val = 0
    error = 0
    cum_error = 0
    sum_derJ1 = np.zeros_like(w2)
    sum_derJ2 = np.zeros_like(w1)
    
    #Your code here
    training_data=np.concatenate((training_data,np.ones((training_data.shape[0],1))),1)
    for i in range(training_data.shape[0]):

        inp = np.array([training_data[i]])
        inp_label = np.array([training_label[i]])
        netJ = np.dot(w1,inp.T)
        Z = sigmoid(netJ)
        Z = np.double(Z)

        Z1 = np.vstack((Z,np.ones((1,1))))
        netL = np.dot(w2,Z1)
        O = sigmoid(netL)
        O = np.double(O)

        error = 0.5 * np.sum(np.subtract(O,training_label[i].T)**2,0)
        cum_error = (cum_error + error)

        derJ1_2 = np.multiply(np.multiply(np.subtract(O,training_label[i].T),O), 1-O)
        derJ1 = np.dot(derJ1_2,Z1.T )
        sum_derJ1 += derJ1

        derJ2_2 = np.array([np.sum(np.multiply(derJ1_2,w2),0)])
        derJ2 = np.dot(np.multiply(np.multiply((1 - Z),Z), derJ2_2.T[0:-1,:]),inp)
        sum_derJ2 += derJ2

    cum_error = (cum_error)/(training_data.shape[0])
    reg_error = cum_error + (lambdaval * (np.sum(np.multiply(w1,w1)) + np.sum(np.multiply(w2,w2)))/(2*(training_data.shape[0])))
    reg_J1 = (sum_derJ1 + (lambdaval * w2))/(training_data.shape[0])
    reg_J2 = (sum_derJ2 + (lambdaval * w1))/(training_data.shape[0])
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((reg_J2.flatten(), reg_J1.flatten()),0)
    weights = np.concatenate((w1.flatten(),w2.flatten()),0)
    obj_val = reg_error
    end_time = int(round(time.time() * 1000))

    cur_dir = os.getcwd()
    cur_dir = os.path.join(cur_dir, "out")
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)
    print("Obj val : ",obj_val, "diff : ",(obj_val-prev_reg) ," completed in  : ",end_time-start_time)
    prev_reg = obj_val
    
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    labels = np.zeros((data.shape[0],1))
    data=np.hstack((data,np.ones((data.shape[0],1))))
    for i in range(data.shape[0]):

        inp = np.array([data[i]])
        netJ = np.dot(w1,inp.T)
        Z = sigmoid(netJ)
        Z = np.double(Z)

        Z1 = np.vstack((Z,[1]))
        netL = np.dot(w2, Z1)
        O = sigmoid(netL)
        O = np.double(O)
        labels[i] = [np.argmax(O,0)]
    
    return labels
    



"""**************Neural Network Script Starts here********************************"""
start_time = int(round(time.time() * 1000))
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = gl_n_hidden

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = gl_lambdaval


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)
cur_dir = os.getcwd()
cur_dir = os.path.join(cur_dir, "out")
if not os.path.exists(cur_dir):
    os.makedirs(cur_dir)


analysis = np.array([])
analysis = np.append(analysis, gl_lambdaval)
analysis = np.append(analysis, gl_n_hidden)

#find the accuracy on Training Dataset
train_label = np.argmax(train_label,2)
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label))) + '%')
np.savetxt(os.path.join(cur_dir,"Train_Results_"+str(int(gl_lambdaval*10))+"_"+str(gl_n_hidden)+".txt"), np.concatenate((train_label,predicted_label),1))
analysis = np.append(analysis,100*np.mean((predicted_label == train_label)))

predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
validation_label = np.argmax(validation_label,2)
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label))) + '%')
np.savetxt(os.path.join(cur_dir,"Validation_Results_"+str(int(gl_lambdaval*10))+"_"+str(gl_n_hidden)+".txt"), np.concatenate((validation_label,predicted_label),1))
analysis = np.append(analysis,100*np.mean((predicted_label == validation_label)))



predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
test_label = np.argmax(test_label,2)
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label))) + '%')
np.savetxt(os.path.join(cur_dir,"Test_Results_"+str(int(gl_lambdaval*10))+"_"+str(gl_n_hidden)+".txt"), np.concatenate((test_label,predicted_label),1))
analysis = np.append(analysis,100*np.mean((predicted_label == test_label)))

weights = np.concatenate((w1.flatten(),w2.flatten()),0)
np.savetxt(os.path.join(cur_dir,"Weights_"+str(int(gl_lambdaval*10))+"_"+str(gl_n_hidden)+".txt"), weights)
np.savetxt(os.path.join(cur_dir,"Analysis_Results_"+str(int(gl_lambdaval*10))+"_"+str(gl_n_hidden)+".txt"), analysis)
end_time = millis = int(round(time.time() * 1000))
print("Total Time : ",(end_time - start_time)//60000," mins ",((end_time - start_time)%60000)/1000," secs")

save_path = (os.path.join(cur_dir,"params_"+str(int(gl_lambdaval*10))+"_"+str(gl_n_hidden)+"_"+str(int(math.floor(np.mean((predicted_label == test_label))*10000)))+".pickle"))
print(save_path)
save_data = {'n_hidden' : gl_n_hidden, 'w1' : w1, 'w2' : w2, 'lambdaval' : gl_lambdaval,}
with open(save_path,'wb') as handle:
    pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


'''
    retest results from saved weights
'''

#nn_params.x = np.loadtxt(os.path.join(cur_dir,"Weights_"+str(int(gl_lambdaval*10))+"_"+str(gl_n_hidden)+".txt"))

data_2 = {}
with open(save_path,'rb') as handle:
    data_2 = pickle.load(handle)

w1 = data_2['w1']
w2 = data_2['w2']

#w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
#w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

print("Retest with saved weights")
predicted_label = nnPredict(w1,w2,train_data)
#train_label = np.argmax(train_label,2)
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)
#validation_label = np.argmax(validation_label,2)
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label))) + '%')

predicted_label = nnPredict(w1,w2,test_data)
#test_label = np.argmax(test_label,2)
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label))) + '%')

