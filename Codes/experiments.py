import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import PercentFormatter
#################################################################################################################
# load data
train_label=np.genfromtxt('train_labels.csv')
test_label=np.genfromtxt('test_labels.csv')
# train without noise
d1=np.genfromtxt('training_data.1  0  0  0.csv').reshape(-1,1)
d2=np.genfromtxt('training_data.0  1  0  0.csv').reshape(-1,1)
d3=np.genfromtxt('training_data.0  0  1  0.csv').reshape(-1,1)
d4=np.genfromtxt('training_data.0  0  0  1.csv',delimiter=',')
d5=np.genfromtxt('training_data.1  1  1  1.csv',delimiter=',')
data=[d1,d2,d3,d4,d5]
# test without noise
t1=np.genfromtxt('test_data.1  0  0  0.csv').reshape(-1,1)
t2=np.genfromtxt('test_data.0  1  0  0.csv').reshape(-1,1)
t3=np.genfromtxt('test_data.0  0  1  0.csv').reshape(-1,1)
t4=np.genfromtxt('test_data.0  0  0  1.csv',delimiter=',')
t5=np.genfromtxt('test_data.1  1  1  1.csv',delimiter=',')
test=[t1,t2,t3,t4,t5]
# with noise
n1=np.genfromtxt('test_noisy_data.1  0  0  0.csv').reshape(-1,1)
n2=np.genfromtxt('test_noisy_data.0  1  0  0.csv').reshape(-1,1)
n3=np.genfromtxt('test_noisy_data.0  0  1  0.csv').reshape(-1,1)
n4=np.genfromtxt('test_noisy_data.0  0  0  1.csv',delimiter=',')
n5=np.genfromtxt('test_noisy_data.1  1  1  1.csv',delimiter=',')

train=scaler.fit_transform(train_noise_crowd[4])
test=scaler.fit_transform(real_test_crowd[4])
model.fit(train,train_label)
pred=model.predict(test)
perf_measure(test_label,pred)

np.savetxt('pred_crowd_noise_SVM_all.txt', pred,delimiter=',',fmt='%i',newline='\r\n')


# real test
real_test_crowd=[n1,n2,n3,n4,n5]
# people speaking noise
test_noise_crowd=[n1,n2,n3,n4,n5]
train_noise_crowd=[n1,n2,n3,n4,n5]
# morning birds car noise
test_noise_morning=[n1,n2,n3,n4,n5]
# people speaking noise
test_noise_typing=[n1,n2,n3,n4,n5]
# train with low noise
train_noise_low=[n1,n2,n3,n4,n5]
# train with medium noise
train_noise_medium=[n1,n2,n3,n4,n5]
# train with high noise
train_noise_high=[n1,n2,n3,n4,n5]
# test with low noise 
test_noise_low=[n1,n2,n3,n4,n5]
# test with medium noise 
test_noise_medium=[n1,n2,n3,n4,n5]
# test with high noise 
test_noise_high=[n1,n2,n3,n4,n5]
#################################################################################################################
# test all models on real noise
SAN_all_1=[]
NAS_all_1=[]
MER_all_1=[]
for i in range(5):
   [speech_as_noise,noise_as_speech,mean_error_rate]=train_test_model(train_noise_low[i],test_noise_crowd[i],rbf)
   SAN_all_1.append(speech_as_noise)
   NAS_all_1.append(noise_as_speech) 
   MER_all_1.append(mean_error_rate) 
plt.style.use('seaborn')
SAN=SAN_all_1
NAS=NAS_all_1
fig, ax = plt.subplots()
plt.style.use('seaborn')
col=['purple','orangered','teal','cyan','black']
plt.scatter(SAN,NAS,alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0],NAS[0]+0.015),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1],NAS[1]+0.015),size=14,ha='center')
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2]-0.045,NAS[2]-0.005),size=14,ha='center')
plt.annotate('MFCC',xy=(SAN[3],NAS[3]),xytext=(SAN[3],NAS[3]+0.015),size=14,ha='center')
plt.annotate('All features',xy=(SAN[4],NAS[4]),xytext=(SAN[4]-0.023,NAS[4]+0.005),size=14,ha='center')
plt.title('SVM trained on low noise data - tested on crowd speaking noise',size=16,y=1.02,x=0.45)
plt.xlim(0,0.23)
plt.ylim(0,0.4)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)
#################################################################################################################
# models
linear=SVC(kernel='linear')
rbf=SVC(kernel='rbf')
#################################################################################################################
# Experiment1: No-noise train, no-noise test
SAN_all_1=[]
NAS_all_1=[]
MER_all_1=[]
for i in range(5):
   [speech_as_noise,noise_as_speech,mean_error_rate]=train_test_model(data[i],test[i],rbf)
   SAN_all_1.append(speech_as_noise)
   NAS_all_1.append(noise_as_speech) 
   MER_all_1.append(mean_error_rate) 
# plot
plt.style.use('seaborn')
SAN=SAN_all_1
NAS=NAS_all_1
fig, ax = plt.subplots()
plt.style.use('seaborn')
col=['purple','orangered','teal','cyan','black']
plt.scatter(SAN,NAS,alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0],NAS[0]+0.0015),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1],NAS[1]+0.002),size=14,ha='center')
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2],NAS[2]+0.002),size=14,ha='center')
plt.annotate('MFCC',xy=(SAN[3],NAS[3]),xytext=(SAN[3],NAS[3]+0.002),size=14,ha='center')
plt.annotate('All features',xy=(SAN[4],NAS[4]),xytext=(SAN[4],NAS[4]-0.0043),size=14,ha='center')
plt.title('Experiment 1:  SVM trained and tested on no noise data',size=16,y=1.02,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0,0.055)
plt.ylim(0,0.05)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)
#################################################################################################################
# Experiment2: No noise train, noise test
# low white noise
SAN_all_2=[]
NAS_all_2=[]
MER_all_2=[]
for i in range(5):
   [speech_as_noise,noise_as_speech,mean_error_rate]=train_test_model(data[i],test_noise_low[i],rbf)
   SAN_all_2.append(speech_as_noise)
   NAS_all_2.append(noise_as_speech)  
   MER_all_1.append(mean_error_rate) 
SAN=SAN_all_2
NAS=NAS_all_2    
fig, ax = plt.subplots()
col=['purple','orangered','teal','cyan','black']
plt.scatter(SAN,NAS,alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0]-0.008,NAS[0]+0.02),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1]-0.03,NAS[1]+0.01),size=14,ha='center')
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2]+0.04,NAS[2]+0.02),size=14,ha='center')
plt.annotate('MFCC',xy=(SAN[3],NAS[3]),xytext=(SAN[3]+0.04,NAS[3]),size=14,ha='center')
plt.annotate('All features',xy=(SAN[4],NAS[4]),xytext=(SAN[4],NAS[4]+0.02),size=14,ha='center')
plt.title('Experiment 2:  SVM trained with no noise data - tested on low noise data',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)
# medium white noise
SAN_all_2=[]
NAS_all_2=[]
MER_all_2=[]
for i in range(5):
   [speech_as_noise,noise_as_speech,mean_error_rate]=train_test_model(data[i],test_noise_medium[i],rbf)
   SAN_all_2.append(speech_as_noise)
   NAS_all_2.append(noise_as_speech)  
   MER_all_1.append(mean_error_rate) 
SAN=SAN_all_2
NAS=NAS_all_2    
fig, ax = plt.subplots()
col=['purple','orangered','teal','cyan','black']
plt.scatter(SAN,NAS,alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0]-0.01,NAS[0]+0.014),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1]+0.025,NAS[1]+0.015),size=14,ha='center')
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2],NAS[2]+0.014),size=14,ha='center')
plt.annotate('MFCC',xy=(SAN[3],NAS[3]),xytext=(SAN[3]+0.024,NAS[3]-0.006),size=14,ha='center')
plt.annotate('All features',xy=(SAN[4],NAS[4]),xytext=(SAN[4],NAS[4]+0.014),size=14,ha='center')
plt.title('Experiment 3:  SVM trained with no noise data - tested on medium noise data',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)
# high white noise
SAN_all_2=[]
NAS_all_2=[]
MER_all_2=[]
for i in range(5):
   [speech_as_noise,noise_as_speech,mean_error_rate]=train_test_model(data[i],test_noise_high[i],rbf)
   SAN_all_2.append(speech_as_noise)
   NAS_all_2.append(noise_as_speech)  
   MER_all_1.append(mean_error_rate) 
SAN=SAN_all_2
NAS=NAS_all_2  
fig, ax = plt.subplots()
col=['purple','orangered','teal','cyan','black']
plt.scatter(SAN,NAS,alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0],NAS[0]+0.014),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1],NAS[1]+0.015),size=14,ha='center')
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2]-0.03,NAS[2]+0.014),size=14,ha='center')
plt.annotate('MFCC',xy=(SAN[3],NAS[3]),xytext=(SAN[3]+0.015,NAS[3]+0.014),size=14,ha='center')
plt.annotate('All features',xy=(SAN[4],NAS[4]),xytext=(SAN[4],NAS[4]+0.014),size=14,ha='center')
plt.title('Experiment 4: SVM trained with no noise data - tested on high noise data',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)
# arbitrary noise
idx=np.arange(5)
SAN=[]
NAS=[]
for i in idx:
   SAN.append(SAN_all_2[i])
   NAS.append(NAS_all_2[i])    
fig, ax = plt.subplots()
col=['purple','orangered','teal','cyan','black']
plt.scatter(SAN,NAS,alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0],NAS[0]+0.014),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1],NAS[1]+0.015),size=14,ha='center')
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2]-0.03,NAS[2]+0.014),size=14,ha='center')
plt.annotate('MFCC',xy=(SAN[3],NAS[3]),xytext=(SAN[3]+0.015,NAS[3]+0.014),size=14,ha='center')
plt.annotate('All features',xy=(SAN[4],NAS[4]),xytext=(SAN[4],NAS[4]+0.014),size=14,ha='center')
plt.title('Experiment 5: feature performance of none-noise traied SVM on high noise test data',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)
#################################################################################################################
# Experiment3: noise train, no noise test
# low noise
SAN_all_3=[]
NAS_all_3=[]
MER_all_3=[]
for i in range(5):
   [speech_as_noise,noise_as_speech,mean_error_rate]=train_test_model(train_noise_low[i],test[i],rbf)
   SAN_all_3.append(speech_as_noise)
   NAS_all_3.append(noise_as_speech)
   MER_all_3.append(mean_error_rate) 
SAN=SAN_all_3
NAS=NAS_all_3   
fig, ax = plt.subplots()
col=['purple','orangered','teal','cyan','black']
plt.scatter(SAN[0:5],NAS[0:5],alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0],NAS[0]+0.02),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1]+0.007,NAS[1]+0.005),size=14,ha='center')   
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2],NAS[2]+0.02),size=14,ha='center')
plt.annotate('MFCC',xy=(SAN[3],NAS[3]),xytext=(SAN[3]+0.0057,NAS[3]-0.005),size=14,ha='center')
plt.annotate('All features',xy=(SAN[4],NAS[4]),xytext=(SAN[4],NAS[4]+0.02),size=14,ha='center')   
plt.title('Experiment 5: SVM trained with low noise data - tested on no noise data',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)
# medium noise
SAN_all_3=[]
NAS_all_3=[]
MER_all_3=[]
for i in range(5):
   [speech_as_noise,noise_as_speech,mean_error_rate]=train_test_model(train_noise_medium[i],test[i],rbf)
   SAN_all_3.append(speech_as_noise)
   NAS_all_3.append(noise_as_speech)
   MER_all_3.append(mean_error_rate) 
SAN=SAN_all_3
NAS=NAS_all_3   
fig, ax = plt.subplots()
col=['purple','orangered','teal','cyan','black']
plt.scatter(SAN[0:5],NAS[0:5],alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0],NAS[0]+0.02),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1],NAS[1]+0.02),size=14,ha='center')   
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2]+0.005,NAS[2]+0.02),size=14,ha='center')
plt.annotate('MFCC',xy=(SAN[3],NAS[3]),xytext=(SAN[3],NAS[3]+0.02),size=14,ha='center')
plt.annotate('All features',xy=(SAN[4],NAS[4]),xytext=(SAN[4],NAS[4]-0.025),size=14,ha='center')   
plt.title('Experiment 6: SVM trained with medium noise data - tested on no noise data',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)
# high noise
SAN_all_3=[]
NAS_all_3=[]
MER_all_3=[]
for i in range(5):
   [speech_as_noise,noise_as_speech,mean_error_rate]=train_test_model(train_noise_high[i],test[i],rbf)
   SAN_all_3.append(speech_as_noise)
   NAS_all_3.append(noise_as_speech)
   MER_all_3.append(mean_error_rate) 
SAN=SAN_all_3
NAS=NAS_all_3   
fig, ax = plt.subplots()
col=['purple','orangered','teal','cyan','black']
plt.scatter(SAN[0:5],NAS[0:5],alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0],NAS[0]+0.02),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1]+0.0058,NAS[1]+0.02),size=14,ha='center')   
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2],NAS[2]+0.02),size=14,ha='center')
plt.annotate('MFCC',xy=(SAN[3],NAS[3]),xytext=(SAN[3],NAS[3]+0.02),size=14,ha='center')
plt.annotate('All features',xy=(SAN[4],NAS[4]),xytext=(SAN[4],NAS[4]+0.02),size=14,ha='center')   
plt.title('Experiment 7:  SVM trained with high noise data - tested on no noise data',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)
#################################################################################################################
# Experiment4: noise train, noise test
# low train, low test
SAN_all_4=[]
NAS_all_4=[]
MER_all_4=[]
for i in range(5):
   [speech_as_noise,noise_as_speech,mean_error_rate]=train_test_model(train_noise_low[i],test_noise_low[i],rbf)
   SAN_all_4.append(speech_as_noise)
   NAS_all_4.append(noise_as_speech)
   MER_all_4.append(mean_error_rate) 
SAN=SAN_all_4
NAS=NAS_all_4   
fig, ax = plt.subplots()
col=['purple','orangered','teal','cyan','black']
plt.scatter(SAN[0:5],NAS[0:5],alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0],NAS[0]+0.0015),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1],NAS[1]+0.0015),size=14,ha='center')   
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2],NAS[2]+0.0015),size=14,ha='center')
plt.annotate('MFCC',xy=(SAN[3],NAS[3]),xytext=(SAN[3],NAS[3]-0.0025),size=14,ha='center')
plt.annotate('All features',xy=(SAN[4],NAS[4]),xytext=(SAN[4],NAS[4]+0.0015),size=14,ha='center')   
plt.title('Experiment 8: SVM trained with low noise data - tested on low noise data',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)
# low train, medium test
SAN_all_4=[]
NAS_all_4=[]
MER_all_4=[]
for i in range(5):
   [speech_as_noise,noise_as_speech,mean_error_rate]=train_test_model(train_noise_low[i],test_noise_medium[i],rbf)
   SAN_all_4.append(speech_as_noise)
   NAS_all_4.append(noise_as_speech)
   MER_all_4.append(mean_error_rate) 
SAN=SAN_all_4
NAS=NAS_all_4   
fig, ax = plt.subplots()
col=['purple','orangered','teal','cyan','black']
plt.scatter(SAN[0:5],NAS[0:5],alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0],NAS[0]+0.015),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1],NAS[1]+0.015),size=14,ha='center')   
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2]-0.03,NAS[2]+0.005),size=14,ha='center')
plt.annotate('MFCC',xy=(SAN[3],NAS[3]),xytext=(SAN[3],NAS[3]+0.015),size=14,ha='center')
plt.annotate('All features',xy=(SAN[4],NAS[4]),xytext=(SAN[4],NAS[4]+0.015),size=14,ha='center')   
plt.title('Experiment 9: SVM trained with low noise data - tested on medium noise data',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)
# low train, high test
SAN_all_4=[]
NAS_all_4=[]
MER_all_4=[]
for i in range(5):
   [speech_as_noise,noise_as_speech,mean_error_rate]=train_test_model(train_noise_low[i],test_noise_high[i],rbf)
   SAN_all_4.append(speech_as_noise)
   NAS_all_4.append(noise_as_speech)
   MER_all_4.append(mean_error_rate) 
SAN=SAN_all_4
NAS=NAS_all_4   
fig, ax = plt.subplots()
col=['purple','orangered','teal','cyan','black']
plt.scatter(SAN[0:5],NAS[0:5],alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0],NAS[0]+0.015),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1],NAS[1]+0.015),size=14,ha='center')   
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2]-0.01,NAS[2]-0.025),size=14,ha='center')
plt.annotate('MFCC',xy=(SAN[3],NAS[3]),xytext=(SAN[3],NAS[3]+0.015),size=14,ha='center')
plt.annotate('All features',xy=(SAN[4],NAS[4]),xytext=(SAN[4],NAS[4]+0.015),size=14,ha='center')   
plt.title('Experiment 10: SVM trained with low noise data - tested on high noise data',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)
# medium train, low test
SAN_all_4=[]
NAS_all_4=[]
MER_all_4=[]
for i in range(5):
   [speech_as_noise,noise_as_speech,mean_error_rate]=train_test_model(train_noise_medium[i],test_noise_low[i],rbf)
   SAN_all_4.append(speech_as_noise)
   NAS_all_4.append(noise_as_speech)
   MER_all_4.append(mean_error_rate) 
SAN=SAN_all_4
NAS=NAS_all_4   
fig, ax = plt.subplots()
col=['purple','orangered','teal','cyan','black']
plt.scatter(SAN[0:5],NAS[0:5],alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0],NAS[0]+0.0015),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1],NAS[1]+0.0015),size=14,ha='center')   
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2],NAS[2]+0.0015),size=14,ha='center')
plt.annotate('MFCC',xy=(SAN[3],NAS[3]),xytext=(SAN[3],NAS[3]+0.0015),size=14,ha='center')
plt.annotate('All features',xy=(SAN[4],NAS[4]),xytext=(SAN[4]-0.005,NAS[4]+0.0015),size=14,ha='center')   
plt.title('Experiment 11: SVM trained with medium noise data - tested on low noise data',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)
# medium train, medium test
SAN_all_4=[]
NAS_all_4=[]
MER_all_4=[]
for i in range(5):
   [speech_as_noise,noise_as_speech,mean_error_rate]=train_test_model(train_noise_medium[i],test_noise_medium[i],rbf)
   SAN_all_4.append(speech_as_noise)
   NAS_all_4.append(noise_as_speech)
   MER_all_4.append(mean_error_rate) 
SAN=SAN_all_4
NAS=NAS_all_4   
fig, ax = plt.subplots()
col=['purple','orangered','teal','cyan','black']
plt.scatter(SAN[0:5],NAS[0:5],alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0],NAS[0]+0.0015),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1]+0.013,NAS[1]-0.0008),size=14,ha='center')   
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2]-0.02,NAS[2]+0.0015),size=14,ha='center')
plt.annotate('MFCC',xy=(SAN[3],NAS[3]),xytext=(SAN[3]+0.014,NAS[3]-0.0005),size=14,ha='center')
plt.annotate('All features',xy=(SAN[4],NAS[4]),xytext=(SAN[4]-0.01,NAS[4]+0.0015),size=14,ha='center')
plt.title('Experiment 12: SVM trained with medium noise data - tested on medium noise data',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)
# medium train, high test
SAN_all_4=[]
NAS_all_4=[]
MER_all_4=[]
for i in range(5):
   [speech_as_noise,noise_as_speech,mean_error_rate]=train_test_model(train_noise_medium[i],test_noise_high[i],rbf)
   SAN_all_4.append(speech_as_noise)
   NAS_all_4.append(noise_as_speech)
   MER_all_4.append(mean_error_rate) 
SAN=SAN_all_4
NAS=NAS_all_4   
fig, ax = plt.subplots()
col=['purple','orangered','teal','cyan','black']
plt.scatter(SAN[0:5],NAS[0:5],alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0],NAS[0]+0.008),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1],NAS[1]+0.008),size=14,ha='center')   
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2],NAS[2]+0.008),size=14,ha='center')
plt.annotate('MFCC',xy=(SAN[3],NAS[3]),xytext=(SAN[3],NAS[3]+0.008),size=14,ha='center')
plt.annotate('All features',xy=(SAN[4],NAS[4]),xytext=(SAN[4],NAS[4]+0.008),size=14,ha='center')  
plt.title('Experiment 13: SVM trained with medium noise data - tested on high noise data',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)
# high train, low test
SAN_all_4=[]
NAS_all_4=[]
MER_all_4=[]
for i in range(5):
   [speech_as_noise,noise_as_speech,mean_error_rate]=train_test_model(train_noise_high[i],test_noise_low[i],rbf)
   SAN_all_4.append(speech_as_noise)
   NAS_all_4.append(noise_as_speech)
   MER_all_4.append(mean_error_rate) 
SAN=SAN_all_4
NAS=NAS_all_4   
fig, ax = plt.subplots()
col=['purple','orangered','teal','cyan','black']
plt.scatter(SAN[0:5],NAS[0:5],alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0]-0.003,NAS[0]+0.02),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1]+0.01,NAS[1]-0.03),size=14,ha='center')   
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2],NAS[2]+0.03),size=14,ha='center')
plt.annotate('MFCC',xy=(SAN[3],NAS[3]),xytext=(SAN[3]-0.01,NAS[3]+0.015),size=14,ha='center')
plt.annotate('All features',xy=(SAN[4],NAS[4]),xytext=(SAN[4]+0.008,NAS[4]+0.02),size=14,ha='center')   
plt.title('Experiment 14: SVM trained with high noise data - tested on low noise data',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)
# high train, medium test
SAN_all_4=[]
NAS_all_4=[]
MER_all_4=[]
for i in range(5):
   [speech_as_noise,noise_as_speech,mean_error_rate]=train_test_model(train_noise_high[i],test_noise_medium[i],rbf)
   SAN_all_4.append(speech_as_noise)
   NAS_all_4.append(noise_as_speech)
   MER_all_4.append(mean_error_rate) 
SAN=SAN_all_4
NAS=NAS_all_4   
fig, ax = plt.subplots()
col=['purple','orangered','teal','cyan','black']
plt.scatter(SAN[0:5],NAS[0:5],alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0]+0.025,NAS[0]),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1]+0.013,NAS[1]+0.015),size=14,ha='center')   
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2]+0.005,NAS[2]+0.02),size=14,ha='center')
plt.annotate('MFCC',xy=(SAN[3],NAS[3]),xytext=(SAN[3]-0.02,NAS[3]+0.01),size=14,ha='center')
plt.annotate('All features',xy=(SAN[4],NAS[4]),xytext=(SAN[4]+0.005,NAS[4]+0.02),size=14,ha='center')  
plt.title('Experiment 15: SVM trained with high noise data - tested on medium noise data',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)
# high train, high test
SAN_all_4=[]
NAS_all_4=[]
MER_all_4=[]
for i in range(5):
   [speech_as_noise,noise_as_speech,mean_error_rate]=train_test_model(train_noise_high[i],test_noise_high[i],rbf)
   SAN_all_4.append(speech_as_noise)
   NAS_all_4.append(noise_as_speech)
   MER_all_4.append(mean_error_rate) 
SAN=SAN_all_4
NAS=NAS_all_4   
fig, ax = plt.subplots()
col=['purple','orangered','teal','cyan','black']
plt.scatter(SAN[0:5],NAS[0:5],alpha=0.7,color=col,marker='o',s=150)
plt.annotate('Zero crossing',xy=(SAN[0],NAS[0]),xytext=(SAN[0],NAS[0]+0.015),size=14,ha='center')
plt.annotate('Energy',xy=(SAN[1],NAS[1]),xytext=(SAN[1],NAS[1]+0.015),size=14,ha='center')   
plt.annotate('One-lag autocorrelation',xy=(SAN[2],NAS[2]),xytext=(SAN[2],NAS[2]+0.015),size=14,ha='center')
plt.annotate('MFCC',xy=(SAN[3],NAS[3]),xytext=(SAN[3]+0.01,NAS[3]+0.015),size=14,ha='center')
plt.annotate('All features',xy=(SAN[4],NAS[4]),xytext=(SAN[4],NAS[4]+0.015),size=14,ha='center') 
plt.title('Experiment 16: SVM trained with high noise data - tested on high noise data',size=16,y=1.05,x=0.45)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Speech as noise error rate',size=13.5)
plt.ylabel('Noise as speech error rate',size=13.5)
#################################################################################################################
# train_test function
scaler=StandardScaler()
def train_test_model(train,test,model):
   train=scaler.fit_transform(train)
   test=scaler.fit_transform(test)
   model.fit(train,train_label)
   pred=model.predict(test)
   [SAN,NAS,MER]=perf_measure(test_label,pred)
   return [SAN,NAS,MER]

# error rates
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
    MER=FN_ER+FP_ER
    return(FN_ER, FP_ER, MER)
    
    
#################################################################################################################  
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


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
    out = ax.contourf(xx, yy, Z, **params,antialiased=True)
    return out

# example decision boundry
scaler=StandardScaler()
X0=scaler.fit_transform(train_noise_medium[3][:400,:]).ravel()
X1=scaler.fit_transform(train_noise_medium[2][:400,:]).ravel()
X = np.column_stack((X0,X1))
y = train_label[0:400]
plt.style.use('seaborn')
model=SVC()
model.fit(X,y)
xx, yy = make_meshgrid(X0, X1) 
df = pd.DataFrame(dict(x=X0, y=X1, label=y))
groups = df.groupby('label')
# plot
fig, ax = plt.subplots(figsize=(12,7))
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
   plot_contours(plt, model, xx, yy, cmap=plt.cm.tab20c, alpha=0.3)
   ax.plot(group.x, group.y, marker='o', linestyle='', ms=8, label=name)
L=ax.legend(loc='center right',fontsize=14)
L.get_texts()[1].set_text('Speech')
L.get_texts()[0].set_text('None-speech')
ax.set_xlabel('MFCC',size=17)
ax.set_ylabel('One-lag autocorrelation',size=17)
ax.set_ylim(-2,2.2)
ax.set_title('SVM with two features on train and test data with medium noise',size=20,y=1.02,x=0.45)











