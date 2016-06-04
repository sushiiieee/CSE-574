import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
from pylab import savefig

count = 0

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix

    # IMPLEMENT THIS METHOD

    combined = np.hstack((X,y))
    combined_ind = np.lexsort(np.transpose(combined)[:3])
    combined = combined[combined_ind]
    combined = np.asarray(np.split(combined, np.where(np.diff(combined[:,2] ))[0]+1))

    means = []
    covmat = []
    csize = []
    for c in combined:
        c = np.delete(c,2,1)
        m = np.mean(c,0)
        means.append(m)
        csize.append(c.shape[0])
        cov = (np.dot((c-m).T,(c-m)) / c.shape[0]) * (c.shape[0]-1)
        covmat.append(cov)

    csize = np.asarray(csize)
    covmat = np.sum(np.asarray(covmat),0)/(np.sum(csize)-csize.shape[0])

    # covmat = np.asarray(covmat)
    # print("lda",covmat)
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD

    combined = np.hstack((X,y))
    combined_ind = np.lexsort(np.transpose(combined)[:3])
    combined = combined[combined_ind]
    combined = np.asarray(np.split(combined, np.where(np.diff(combined[:,2] ))[0]+1))

    means = []
    covmat = []
    for c in combined:
        c = np.delete(c,2,1)
        m = np.mean(c,0)
        means.append(m)
        cov = np.dot((c-m).T,(c-m)) / c.shape[0]
        covmat.append(cov)

    covmats = np.asarray(covmat)
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    inv_cov = inv(covmat)
    ypred = np.array([])
    for x in Xtest:
        dist = np.array([])
        for index, m in enumerate(means):
            dist = np.append(dist, np.dot(np.dot((x-m),inv_cov),(x-m).T))
        if np.shape(ypred)[0] == 0:
            ypred = np.array([np.argmin(dist)+1])
        else:
            ypred = np.vstack((ypred,np.array([np.argmin(dist)+1])))


    acc = np.mean((ypred==ytest).astype(float))*100
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    inv_cov = inv(covmats)
    # print(covmats, inv_cov)
    ypred = np.array([])
    for x in Xtest:
        dist = np.array([])
        for index, m in enumerate(means):
            dist = np.append(dist, np.dot(np.dot((x-m),inv_cov[index]),(x-m).T))
        if np.shape(ypred)[0] == 0:
            ypred = np.array([np.argmin(dist)+1])
        else:
            ypred = np.vstack((ypred,np.array([np.argmin(dist)+1])))


    acc = np.mean((ypred==ytest).astype(float))*100
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1
    w=np.dot(inv(np.dot(X.T,X)),np.dot(X.T,y))
    # IMPLEMENT THIS METHOD
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1
    # product=np.dot(X.T,X)
    # prod=np.dot(X.T,y)
    identity=np.identity(X.shape[1])
    # product1=(lambd*X.shape[0]*identity)
    # sumation=product+product1
    # part1=np.linalg.inv(sumation)
    # w=np.dot(part1,prod)

    w=np.dot(inv(np.dot(X.T,X) + (lambd * identity)),np.dot(X.T,y))

    # IMPLEMENT THIS METHOD

    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse

    predict=np.dot(Xtest,w)
    sq = np.dot((ytest-predict).T, (ytest-predict))
    rmse = np.sqrt(np.mean(sq))

    # IMPLEMENT THIS METHOD
    return rmse

def regressionObjVal(w, X, y, lambd):
    '''
    error = 0
    w_transpose = np.transpose(w)
    X_transpose = np.transpose(X)
    y_transpose = np.transpose(y)
    temp_1 = np.subtract(y_transpose, np.dot(w_transpose, X_transpose))
    temp_2 = np.multiply(temp_1, temp_1)
    sum = np.sum(temp_2)
    N = X.shape[0]
    val1 = sum
    temp_3 = np.dot(w_transpose, w)
    val2 = np.multiply(lambd, temp_3)
    # print val1
    # print val2
    error = (val1 + val2) / 2
    error_grads = np.add(np.subtract(np.multiply(np.multiply(lambd, N), w), np.dot(y_transpose, X)),
                         np.dot(w_transpose, np.dot(X_transpose, X)))
    error_grad = np.squeeze(np.asarray(error_grads)) / N

    '''
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda
    global count
    error = 0
    predict1=np.dot(X,w.T)
    np.expand_dims(y,0)
    sq = np.dot((y-predict1).T, (y-predict1))
    error1=(np.mean(sq))
    error12=np.dot(w.T,w)
    error2=(lambd*error12)/2
    error=error1+error2

    error_grad = 0
    one=np.dot(X.T,X)
    one1=np.dot(w,one)
    two=np.squeeze(np.dot(X.T,y))
    three=lambd*w

    error_grad1=one1-two.T+three
    error_grad=np.squeeze(error_grad1)


    count = count + 1

    # IMPLEMENT THIS METHOD
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))
    # IMPLEMENT THIS METHOD
    Xd=np.zeros((x.shape[0],p+1))
    for i in range(0,p+1) :
        Xd[:,i] = pow(x,i)
    return Xd

# Main script

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# if sys.version_info.major == 2:
#     X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
# else:
#     X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# print(y)
# print(ytest)

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

plt.figure(1)
zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
savefig("p1.png")
# plt.show()

plt.figure(2)
zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
savefig("p2.png")
# plt.show()

# Problem 2

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# print("Diabetes : ")
# print(X)
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 0.1, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1

######edit
# print(lambdas,rmses3)
# print("shape")
# print(lambdas.shape)
# print("shape of rmses")
# print(rmses3.shape)
# print(lambdas,rmses3)
# print(lambdas[np.argmin(rmses3)])
plt.figure(3)
plt.plot(lambdas,rmses3)
savefig("p3.png")
#plt.show()
print("min p3 : " + str(lambdas[np.argmin(rmses3)]))

# Problem 4
k = 101
lambdas = np.linspace(0, 0.1, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    #print("Executed "+str(i) +" for : "+str(count)+" iters")
    count = 0
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.figure(4)
plt.plot(lambdas,rmses4)
savefig("p4.png")
#plt.show()

print("min p4 : " + str(lambdas[np.argmin(rmses4)]))


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.figure(5)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
savefig("p5.png")
