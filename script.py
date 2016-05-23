import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


#%matplotlib inline

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    #y=y[1:10]
    #X=X[1:10]
#    print "AAAAAAAAA"
    #print y
    
    

    uni=np.unique(y)
    means_l=[]
    cov_l=[]
    for each in np.unique(y):
        sub_temp=[]
        for k in np.where(y==each)[0]:
            sub_temp.append(X[k])
        mean_temp=np.mean(sub_temp,axis=0)

        var_sub=sub_temp-mean_temp
        a=np.cov(var_sub,rowvar =0)
        cov_l.append(a)
        
        means_l.append(mean_temp)
        
    means=np.asarray(means_l)
    covmat=cov_l[0]

    mean_all=np.mean(X,axis=0)
    var_all=X-mean_all
    covmat=np.cov(var_all,rowvar =0)
        
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
        #y=y[1:10]
    #X=X[1:10]
#    print "AAAAAAAAAAAA"
    uni=np.unique(y)
    means_l=[]
    cov_l=[]
    for each in np.unique(y):
        sub_temp=[]
        for k in np.where(y==each)[0]:
            sub_temp.append(X[k])
        mean_temp=np.mean(sub_temp,axis=0)

        var_sub=sub_temp-mean_temp
        
        a=np.cov(var_sub,rowvar =0)
        cov_l.append(a)
        
        means_l.append(mean_temp)
    means=np.asarray(means_l)
    covmats=cov_l
    
  
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    #
    #print means.shape
    #print covmat.shape
    #print Xtest.shape
    #print ytest.shape

    
    output_for_class=[]

    for each in range(means.shape[0]):
        inv_covar = np.linalg.inv(covmat)
        delterminent_covar = np.linalg.det(covmat)
        outlist=[]
        D=inv_covar.shape[0]
        
        for x in Xtest:            
            b=(np.sqrt(delterminent_covar)*(np.power(np.pi*2,D/1)))
            p=np.dot((x - means[each]).reshape(1,-1),inv_covar)
            q=np.dot(p,(x - means[each]).reshape(-1,1))
            

            temp_pred_per_t=np.exp(-0.5*q[0][0])/b;        
            outlist.append(temp_pred_per_t)
            outlist_asarray=np.asarray(outlist,dtype='float32')

        output_for_class.append(outlist_asarray.flatten())#1 row
    our_outputs=np.asarray(output_for_class,dtype='float32')
    
    
    
    count=0
    pred1=[]
    for y in range(len(ytest)):        
        acc1=np.argmax(our_outputs[:,y])+1
        pred1.append(acc1)
    
    pred=np.asarray(pred1,dtype='float32')
    
    pred=pred.reshape(-1,1)
    acc=100*np.mean((pred == ytest).astype(float))
    ypred=pred
    
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    
    output_for_class=[]

    for each in range(means.shape[0]):
        inv_covar = np.linalg.inv(covmats[each])
        delterminent_covar = np.linalg.det(covmats[each])
        outlist=[]
        D=inv_covar.shape[0]
        for x in Xtest:
            b=(np.sqrt(delterminent_covar)*(np.power(np.pi*2,D/2)))
            p=np.dot((x - means[each]).reshape(1,-1),inv_covar)
            q=np.dot(p,(x - means[each]).reshape(-1,1))
            

            temp_pred_per_t=np.exp(-0.5*q[0][0])/b;        
            outlist.append(temp_pred_per_t)
            outlist_asarray=np.asarray(outlist,dtype='float32')            
        output_for_class.append(outlist_asarray.flatten())#1 row

    our_outputs=np.asarray(output_for_class,dtype='float32')    
    count=0

    pred1=[]
    for y in range(len(ytest)):        
        acc1=np.argmax(our_outputs[:,y])+1
        pred1.append(acc1)
    
    pred=np.asarray(pred1,dtype='float32')
    pred=pred.reshape(-1,1)
    acc=100*np.mean((pred == ytest).astype(float))
    ypred=pred
    return acc,ypred
    
def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD      
    ###
    #formula in handout 4.3 learning parameters
    ###                                        
    
    w=np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))

    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    
    #6.1 topic formula                                                   
    iden=np.identity(X.shape[1])
    lambdI=lambd*iden
    w=np.dot(np.linalg.inv(np.add(np.dot(X.T,X),lambdI)),np.dot(X.T,y))
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD
    #print ytest.shape
    #print Xtest.shape
    #print w.shape
    #print np.dot(Xtest,w).shape

    #print np.sum(np.square(ytest-np.dot(Xtest,w)))
    #print Xtest.shape[0]
    #print np.sum(np.square(ytest-np.dot(Xtest,w)))/Xtest.shape[0]

    rmse=np.sqrt(np.sum(np.square(ytest-np.dot(Xtest,w)))/Xtest.shape[0])
    

    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD                                             
    ##error in scanned notes
    ##https://www.youtube.com/watch?v=gvb49vLEk0k 29:35
    #print y.shape
    #print X.shape
    #print w.shape
    #print w.reshape(-1,1).shape
#    print "yyyhyhyhyhyhyyhyhyhyhyh"

   
    w=w.reshape(-1,1)
    error1=np.sum(np.square(y-np.dot(X,w)))/2
    error2=(lambd*np.dot(w.T,w))/2
    error=error1+error2
    error_grad1=np.dot(np.dot(w.T,X.T),X)
    error_grad1=error_grad1.T

    error_grad2=np.dot(X.T,y)

    error_grad3=lambd*w

    error_grad=error_grad1-error_grad2+error_grad3
    error_grad=error_grad.reshape(-1)



    
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    
    total=np.empty((0,p+1), int)
    
    for eachx in x:
        
        row=[]
        for i in range(p+1):
            tt=np.power(eachx,i)
            row.append(tt)
        
        rowa=np.asarray(row,dtype='float32')
        #print 'here'
        #print rowa
        total=np.vstack((total,rowa))
        #print total


    totala=np.asarray(total,dtype='float32')
    
    #print totala
    #print totala.shape
    #
    #print x
    #print p
    Xd=totala
    #print Xd.shape
    
    return Xd

# Main script
#print "PROBLEM 1"
# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
#    print len(X)
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

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

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
#
#plt.show()
# Problem 2
#print "PROBLEM 2"
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')



# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)
#mle_train= testOLERegression(w,X,y)


w_i = learnOLERegression(X_i,y)
#print "mean of w_i from ole "+str(np.mean(w_i))

mle_i = testOLERegression(w_i,Xtest_i,ytest)
#mle_i_train = testOLERegression(w_i,X_i,y)


print('RMSE without intercept(test data) '+str(mle))
print('RMSE with intercept(test data) '+str(mle_i))

#print('RMSE without intercept(train data) '+str(mle_train))
#print('RMSE with intercept(train data) '+str(mle_i_train))



# Problem 3
#print "PROBLEM 3"
k = 101
lambdas = np.linspace(0, 1, num=k)

i = 0
rmses3 = np.zeros((k,1))
#rmses3_org_train = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)    
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)    
#    rmses3_org_train[i] = testOLERegression(w_l,X_i,y)
    i = i + 1
plt.plot(lambdas,rmses3)
#plt.plot(lambdas,rmses3,label='test')
#plt.plot(lambdas,rmses3_org_train,label='train')
#plt.legend(loc='upper right')
#plt.show()
##extra code

#
#print "Min of rmse3(with intercept-train data) at opt lambda "+str(rmses3_org_train[np.argmin(rmses3)])
#
#lambda_opt1 = lambdas[np.argmin(rmses3)]
#print "Min of rmse3(with intercept-test data) at opt lambda "+str(rmses3[np.argmin(rmses3)])
#print "Opt Lambda "+ str(lambda_opt1)
#
#w_l = learnRidgeRegression(X_i,y,lambda_opt1) 
#print "mean of w_l from ridge using opt lambda "+str(np.mean(w_l))
#
#plt.plot(w_i,label='Weights OLE reg')
#plt.plot(w_l,label='Weights Ridge')
#plt.legend(loc='upper right')
#plt.show()

# Problem 4
#print "PROBLEM 4"
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
#rmses4_org_train = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))

for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
#    rmses4_org_train[i] = testOLERegression(w_l,X_i,y)
    
    i = i + 1
plt.plot(lambdas,rmses4)
#plt.plot(lambdas,rmses4,label='test')
#plt.plot(lambdas,rmses4_org_train,label='train')
#
#plt.legend(loc='upper right')
#plt.show()
#lambda_opt = lambdas[np.argmin(rmses4)]
#print "lambda_opt ridge gradient " +str(lambda_opt)
#
#print "(Gradient Descent) Min of rmse3(with intercept-train data) at opt lambda "+str(rmses4_org_train[np.argmin(rmses4)])
#
#
#print "(Gradient Descent) Min of rmse3(with intercept-test data) at opt lambda "+str(rmses4[np.argmin(rmses4)])
#
#
#w_l = learnRidgeRegression(X_i,y,lambda_opt) 
#print "(Gradient Descent) mean of w_l from ridge using opt lambda "+str(np.mean(w_l))
#
#plt.plot(w_l,label='Weights Ridge(Gradient Descent)')
#plt.legend(loc='upper right')
#plt.show()

# Problem 5
#print "PROBLEM 5"
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
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
#plt.legend(('No Regularization(lambdal=0) on testing','Regularization(lambdal=opt) on testing','No Regularization(lambdal=0) on training','Regularization(lambdal=opt) on training'))


#training data
#pmax = 7
#
#
#rmses6 = np.zeros((pmax,2))
#for p in range(pmax):
#    Xd = mapNonLinear(X[:,2],p)
#    Xdtest = mapNonLinear(Xtest[:,2],p)
#    w_d1 = learnRidgeRegression(Xd,y,0)
#    rmses6[p,0] = testOLERegression(w_d1,Xd,y)
#    w_d2 = learnRidgeRegression(Xd,y,lambda_opt1)
#    rmses6[p,1] = testOLERegression(w_d2,Xd,y)
#plt.plot(range(pmax),rmses6)
#plt.legend(('No Regularization(lambdal=0) on testing','Regularization(lambdal=opt) on testing','No Regularization(lambdal=0) on training','Regularization(lambdal=opt) on training'))
#plt.show()
#print "test 0 lambda"
#print rmses5[:,0]
#print "min index: "+ str(np.argmin(rmses5[:,0]))
#print "test opt lambda"
#print rmses5[:,1]
#print "min index: "+ str(np.argmin(rmses5[:,1]))
#print "train 0 lambda"
#print rmses6[:,0]
#print "min index: "+ str(np.argmin(rmses6[:,0]))
#print "train opt lambda"
#print rmses6[:,0]
#print "min index: "+ str(np.argmin(rmses6[:,1]))
#
#
