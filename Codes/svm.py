import numpy as np
import matplotlib.pyplot as plt
import seaborn as plot
import pandas as pd
from sklearn.svm import SVC
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import PercentFormatter

train_label=np.genfromtxt('train_labels.csv')
test_label=np.genfromtxt('test_labels.csv')

# train without noise
d1=np.genfromtxt('training_data.1  0  0  0.csv')
d2=np.genfromtxt('training_data.0  1  0  0.csv')
d3=np.genfromtxt('training_data.0  0  1  0.csv')
d4=np.genfromtxt('training_data.0  0  0  1.csv')
d5=np.genfromtxt('training_data.1  1  0  0.csv',delimiter=',')
d6=np.genfromtxt('training_data.1  0  1  0.csv',delimiter=',')
d7=np.genfromtxt('training_data.1  0  0  1.csv',delimiter=',')
d8=np.genfromtxt('training_data.0  1  1  0.csv',delimiter=',')
d9=np.genfromtxt('training_data.0  1  0  1.csv',delimiter=',')
d10=np.genfromtxt('training_data.0  0  1  1.csv',delimiter=',')
d11=np.genfromtxt('training_data.1  1  1  0.csv',delimiter=',')
d12=np.genfromtxt('training_data.1  1  0  1.csv',delimiter=',')
d13=np.genfromtxt('training_data.1  0  1  1.csv',delimiter=',')
d14=np.genfromtxt('training_data.0  1  1  1.csv',delimiter=',')
d15=np.genfromtxt('training_data.1  1  1  1.csv',delimiter=',')

d1=d1.reshape(-1,1)
d2=d2.reshape(-1,1)
d3=d3.reshape(-1,1)
d4=d4.reshape(-1,1)

# train with noise
n1_d1=np.genfromtxt('test_noisy_data.1  0  0  0.csv')
n1_d2=np.genfromtxt('test_noisy_data.0  1  0  0.csv')
n1_d3=np.genfromtxt('test_noisy_data.0  0  1  0.csv')
n1_d4=np.genfromtxt('test_noisy_data.0  0  0  1.csv')
n1_d5=np.genfromtxt('test_noisy_data.1  1  0  0.csv',delimiter=',')
n1_d6=np.genfromtxt('test_noisy_data.1  0  1  0.csv',delimiter=',')
n1_d7=np.genfromtxt('test_noisy_data.1  0  0  1.csv',delimiter=',')
n1_d8=np.genfromtxt('test_noisy_data.0  1  1  0.csv',delimiter=',')
n1_d9=np.genfromtxt('test_noisy_data.0  1  0  1.csv',delimiter=',')
n1_d10=np.genfromtxt('test_noisy_data.0  0  1  1.csv',delimiter=',')
n1_d11=np.genfromtxt('test_noisy_data.1  1  1  0.csv',delimiter=',')
n1_d12=np.genfromtxt('test_noisy_data.1  1  0  1.csv',delimiter=',')
n1_d13=np.genfromtxt('test_noisy_data.1  0  1  1.csv',delimiter=',')
n1_d14=np.genfromtxt('test_noisy_data.0  1  1  1.csv',delimiter=',')
n1_d15=np.genfromtxt('test_noisy_data.1  1  1  1.csv',delimiter=',')

n1_d1=n1_d1.reshape(-1,1)
n1_d2=n1_d2.reshape(-1,1)
n1_d3=n1_d3.reshape(-1,1)
n1_d4=n1_d4.reshape(-1,1)

# test without noise
t1=np.genfromtxt('test_data.1  0  0  0.csv')
t2=np.genfromtxt('test_data.0  1  0  0.csv')
t3=np.genfromtxt('test_data.0  0  1  0.csv')
t4=np.genfromtxt('test_data.0  0  0  1.csv')
t5=np.genfromtxt('test_data.1  1  0  0.csv',delimiter=',')
t6=np.genfromtxt('test_data.1  0  1  0.csv',delimiter=',')
t7=np.genfromtxt('test_data.1  0  0  1.csv',delimiter=',')
t8=np.genfromtxt('test_data.0  1  1  0.csv',delimiter=',')
t9=np.genfromtxt('test_data.0  1  0  1.csv',delimiter=',')
t10=np.genfromtxt('test_data.0  0  1  1.csv',delimiter=',')
t11=np.genfromtxt('test_data.1  1  1  0.csv',delimiter=',')
t12=np.genfromtxt('test_data.1  1  0  1.csv',delimiter=',')
t13=np.genfromtxt('test_data.1  0  1  1.csv',delimiter=',')
t14=np.genfromtxt('test_data.0  1  1  1.csv',delimiter=',')
t15=np.genfromtxt('test_data.1  1  1  1.csv',delimiter=',')

t1=t1.reshape(-1,1)
t2=t2.reshape(-1,1)
t3=t3.reshape(-1,1)
t4=t4.reshape(-1,1)

# test with noise
n1_t1=np.genfromtxt('test_noisy_data.1  0  0  0.csv')
n1_t2=np.genfromtxt('test_noisy_data.0  1  0  0.csv')
n1_t3=np.genfromtxt('test_noisy_data.0  0  1  0.csv')
n1_t4=np.genfromtxt('test_noisy_data.0  0  0  1.csv')
n1_t5=np.genfromtxt('test_noisy_data.1  1  0  0.csv',delimiter=',')
n1_t6=np.genfromtxt('test_noisy_data.1  0  1  0.csv',delimiter=',')
n1_t7=np.genfromtxt('test_noisy_data.1  0  0  1.csv',delimiter=',')
n1_t8=np.genfromtxt('test_noisy_data.0  1  1  0.csv',delimiter=',')
n1_t9=np.genfromtxt('test_noisy_data.0  1  0  1.csv',delimiter=',')
n1_t10=np.genfromtxt('test_noisy_data.0  0  1  1.csv',delimiter=',')
n1_t11=np.genfromtxt('test_noisy_data.1  1  1  0.csv',delimiter=',')
n1_t12=np.genfromtxt('test_noisy_data.1  1  0  1.csv',delimiter=',')
n1_t13=np.genfromtxt('test_noisy_data.1  0  1  1.csv',delimiter=',')
n1_t14=np.genfromtxt('test_noisy_data.0  1  1  1.csv',delimiter=',')
n1_t15=np.genfromtxt('test_noisy_data.1  1  1  1.csv',delimiter=',')

n1_t1=n1_t1.reshape(-1,1)
n1_t2=n1_t2.reshape(-1,1)
n1_t3=n1_t3.reshape(-1,1)
n1_t4=n1_t4.reshape(-1,1)

data=[d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15]
test=[t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15]
noisy1_test=[n1_t1,n1_t2,n1_t3,n1_t4,n1_t5,n1_t6,n1_t7,n1_t8,n1_t9,n1_t10,n1_t11,n1_t12,n1_t13,n1_t14,n1_t15]
noisy1_train=[n1_d1,n1_d2,n1_d3,n1_d4,n1_d5,n1_d6,n1_d7,n1_d8,n1_d9,n1_d10,n1_d11,n1_d12,n1_d13,n1_d14,n1_d15]

# SVM models
linear=SVC(kernel='linear')
rbf=SVC(kernel='rbf')

#################################################################################################################
# Experiment1: No-noise train, no-noise test
SAN_all_1=[]
NAS_all_1=[]
for i in range(15):
   [speech_as_noise,noise_as_speech]=train_test_model(data[i],test[i],rbf)
   SAN_all_1.append(speech_as_noise)
   NAS_all_1.append(noise_as_speech)
   
# plotting  
idx=[0,1,2,3,14]
SAN=[]
NAS=[]
for i in idx:
   SAN.append(SAN_all_1[i])
   NAS.append(NAS_all_1[i])     
fig, ax = plt.subplots()
plt.style.use('seaborn')
col=['purple','orangered','teal','cyan','black']
plt.scatter(SAN,NAS,alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0],NAS[0]+0.0015),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1],NAS[1]+0.002),size=14,ha='center')
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2],NAS[2]+0.002),size=14,ha='center')
plt.annotate('Cepstrum fundament estimate',xy=(SAN[3],NAS[3]),xytext=(SAN[3]+0.013,NAS[3]-0.0005),size=14,ha='center')
plt.annotate('All features',xy=(SAN[4],NAS[4]),xytext=(SAN[4]+0.004,NAS[4]+0.002),size=14,ha='center')
plt.title('Figure 1: feature performance on none-noise train and test data with SVM',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0,0.055)
plt.ylim(0,0.05)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)
   
# plotting  
SAN=SAN_all_1[4:16]
NAS=NAS_all_1[4:16]  
fig, ax = plt.subplots()
plt.style.use('seaborn')
end=11
idx=['ZCR+E','ZCR+OLA','ZCR+CFE','E+OLA','E+CFE','OLA+CFE','ZCR+E+OLA','ZCR+E+CFE','ZCR+OLA+CFE','E+OLA+CFE','all']
plt.scatter(SAN[0:end],NAS[0:end],alpha=0.3,color='purple',marker='o',s=150)
for i,label in enumerate(idx):
   plt.annotate(label,xy=(SAN[i],NAS[i]),xytext=(SAN[i]+0.001,NAS[i]+0.001),size=14,ha='center',alpha=0.6)
plt.title('Figure 2: Multiple features performance on none-noise train and test data with SVM',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0,0.05)
plt.ylim(0,0.05)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)

#################################################################################################################
# Experiment2: No-noise train, noise test
SAN_all_2=[]
NAS_all_2=[]
for i in range(15):
   [speech_as_noise,noise_as_speech]=train_test_model(data[i],noisy1_test[i],rbf)
   SAN_all_2.append(speech_as_noise)
   NAS_all_2.append(noise_as_speech)

# plotting
idx=[0,1,2,3,14]
SAN=[]
NAS=[]
for i in idx:
   SAN.append(SAN_all_2[i])
   NAS.append(NAS_all_2[i])    
fig, ax = plt.subplots()
plt.style.use('seaborn')
col=['purple','orangered','teal','cyan','black']
plt.scatter(SAN,NAS,alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0]+0.002,NAS[0]+0.01),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1]+0.001,NAS[1]+0.01),size=14,ha='center')
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2]-0.03,NAS[2]+0.01),size=14,ha='center')
plt.annotate('Cepstrum fundament estimate',xy=(SAN[3],NAS[3]),xytext=(SAN[3]+0.013,NAS[3]+0.01),size=14,ha='center')
plt.annotate('All features',xy=(SAN[4],NAS[4]),xytext=(SAN[4],NAS[4]+0.01),size=14,ha='center')
plt.title('Figure 2: feature performance of none-noise traied SVM on noisy test data',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)

# plotting  
SAN=SAN_all_2[4:16]
NAS=NAS_all_2[4:16]  
fig, ax = plt.subplots()
plt.style.use('seaborn')
end=11
idx=['ZCR+E','ZCR+OLA','ZCR+CFE','E+OLA','E+CFE','OLA+CFE','ZCR+E+OLA','ZCR+E+CFE','ZCR+OLA+CFE','E+OLA+CFE','all']
plt.scatter(SAN[0:end],NAS[0:end],alpha=0.3,color='purple',marker='o',s=150)
for i,label in enumerate(idx):
   plt.annotate(label,xy=(SAN[i],NAS[i]),xytext=(SAN[i]+0.004,NAS[i]+0.01),size=14,ha='center',alpha=0.6)
plt.title('Figure 4: Multiple features performance on noisy test data with none-noise trained SVM',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)

#################################################################################################################
# Experiment3: Noisy train, no-noise test
SAN_all_3=[]
NAS_all_3=[]
for i in range(15):
   [speech_as_noise,noise_as_speech]=train_test_model(noisy1_train[i],test[i],rbf)
   SAN_all_3.append(speech_as_noise)
   NAS_all_3.append(noise_as_speech)

# plotting
SAN=SAN_all_3
NAS=NAS_all_3
fig, ax = plt.subplots()
plt.style.use('seaborn')
end=4
col=['purple','orangered','teal','cyan']
plt.scatter(SAN[0:end],NAS[0:end],alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0],NAS[0]+0.02),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1]+0.007,NAS[1]+0.002),size=14,ha='center')   
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2]-0.005,NAS[2]+0.02),size=14,ha='center')
plt.annotate('Cepstrum fundament estimate',xy=(SAN[3],NAS[3]),xytext=(SAN[3]-0.005,NAS[3]+0.02),size=14,ha='center')
plt.title('Figure 5: Single feature performance on none-noisy test data with noisy traied SVM',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)

# plotting  
SAN=SAN_all_3[4:16]
NAS=NAS_all_3[4:16]  
fig, ax = plt.subplots()
plt.style.use('seaborn')
end=11
idx=['ZCR+E','ZCR+OLA','ZCR+CFE','E+OLA','E+CFE','OLA+CFE','ZCR+E+OLA','ZCR+E+CFE','ZCR+OLA+CFE','E+OLA+CFE','all']
plt.scatter(SAN[0:end],NAS[0:end],alpha=0.3,color='purple',marker='o',s=150)
for i,label in enumerate(idx):
   plt.annotate(label,xy=(SAN[i],NAS[i]),xytext=(SAN[i],NAS[i]+0.01),size=14,ha='center',alpha=0.6)
plt.title('Figure 6: Multiple features performance on none-noise test data with noisy traied SVM',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)

#################################################################################################################
# Experiment4: noise train, noise test
SAN_all_4=[]
NAS_all_4=[]
for i in range(15):
   [speech_as_noise,noise_as_speech]=train_test_model(noisy1_train[i],noisy1_test[i],rbf)
   SAN_all_4.append(speech_as_noise)
   NAS_all_4.append(noise_as_speech)

# plotting
SAN=SAN_all_4
NAS=NAS_all_4
fig, ax = plt.subplots()
plt.style.use('seaborn')
end=4
col=['purple','orangered','teal','cyan']
plt.scatter(SAN[0:end],NAS[0:end],alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0]+0.015,NAS[0]-0.02),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1],NAS[1]+0.013),size=14,ha='center')   
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2],NAS[2]-0.025),size=14,ha='center')
plt.annotate('Cepstrum fundament estimate',xy=(SAN[3],NAS[3]),xytext=(SAN[3]-0.02,NAS[3]+0.01),size=14,ha='center')
plt.title('Figure 11: Single feature performance on noisy trained SVM tested with louder noisy data',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)

# plotting  
SAN=SAN_all_4[4:16]
NAS=NAS_all_4[4:16]  
fig, ax = plt.subplots()
plt.style.use('seaborn')
end=11
idx=['ZCR+E','ZCR+OLA','ZCR+CFE','E+OLA','E+CFE','OLA+CFE','ZCR+E+OLA','ZCR+E+CFE','ZCR+OLA+CFE','E+OLA+CFE','all']
plt.scatter(SAN[0:end],NAS[0:end],alpha=0.3,color='purple',marker='o',s=150)
for i,label in enumerate(idx):
   plt.annotate(label,xy=(SAN[i],NAS[i]),xytext=(SAN[i],NAS[i]+0.01),size=14,ha='center',alpha=0.6)
plt.title('Figure 12: Multiple features performance on noisy trained SVM and tested with louder noisy data',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)


# train different models with different feature sets
scaler=StandardScaler()
def train_test_model(train,test,model):
   train=scaler.fit_transform(train)
   test=scaler.fit_transform(test)
   model.fit(train,train_label)
   pred=model.predict(test)
   [SAN,NAS]=perf_measure(test_label,pred)
   return [SAN,NAS]

# false positive and false negative rates
def perf_measure(y_actual, y_hat):
    FP = 0
    FN = 0
    for i in range(len(y_hat)): 
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    for i in range(len(y_hat)): 
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1 
    FN_ER=FN/len(y_actual)
    FP_ER=FP/len(y_actual)
    return(FN_ER, FP_ER)





# graph decision boundry
X = d6
y = train_label

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

# graph functions
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out



   






























