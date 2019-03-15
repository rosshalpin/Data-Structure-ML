# -*- coding: utf-8 -*-
"""
Code from Jupyter notebook, not guaranteed to run correctly outside of Jupyter enviroment.

Created on Fri Nov 30 11:16:40 2018

@author: Ross Halpin

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from contrastive import CPCA
from sklearn.metrics import classification_report
from sklearn.neighbors.kde import KernelDensity
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.regularizers import l2
import matplotlib.pyplot as plt

run_kde = False
n_features = 2

# loading train.txt value
data = pd.read_csv('train.txt', sep=' ', header=None) # pandas used as it automatically parses scientific notation
xy_data = data.loc[:,1:].values # get as array
# splitting data into train and test data 80:20
X_train, X_test, y_train, y_test = train_test_split( xy_data[:,:8], xy_data[:,8], test_size=0.20, shuffle=False)

classification_One = X_train[y_train[:] == 1, :8] # all points labeled 1 by KNN classification 
#classification_Zero = X_train[y_train[:] == 0, :8] # all points labeled 0 by KNN classification 

# performing contrastive principal componenet analysis
mdl = CPCA(n_components = n_features)
CPCA_train = mdl.fit_transform(X_train, classification_One, alpha_selection='manual', alpha_value=12)
CPCA_test = mdl.fit_transform(X_test, classification_One[:10000,:], alpha_selection='manual', alpha_value=12)
CPCA_test[:,1] = np.negative(CPCA_test[:,1]) # inverted y axis as data loaded inverted

# calculate kernel denisty estimations(takes a while to complete)
if run_kde == True:
    kde_train = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(CPCA_train)
    result_train = kde_train.score_samples(CPCA_train)
    np.savetxt('classDensity_train.txt', result_train)
    
    kde_test = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(CPCA_test)
    result_test = kde_test.score_samples(CPCA_test)
    np.savetxt('classDensity_test.txt', result_test)

classDensity_train = np.loadtxt('classDensity_train.txt')
classDensity_test = np.loadtxt('classDensity_test.txt')

# assembling training feature set
train_ = np.zeros((len(CPCA_train), n_features+9))
train_[:,:n_features] = CPCA_train # cPCA 2D reduced feature set
train_[:,n_features:n_features+8] = X_train # original 8 dimensonal feature set
train_[:,n_features+8] = classDensity_train # Results of KDE
X_train = train_

# assembling test feature set
test_ = np.zeros((len(CPCA_test), n_features+9))
test_[:,:n_features] = CPCA_test # cPCA 2D reduced feature set
test_[:,n_features:n_features+8] = X_test # original 8 dimensonal feature set
test_[:,n_features+8] = classDensity_test # Results of KDE

# Splitting test set into test and validation 50:50
X_valid = test_[:int(len(y_test)/2), :]
X_test = test_[int(len(y_test)/2):, :]
y_valid = y_test[:int(len(y_test)/2)]
y_test = y_test[int(len(y_test)/2):]

# Sequentail Model https://github.com/Msanjayds/Keras/blob/master/Classification_Model_using_Keras.ipynb
model = Sequential()
# Hidden Layer-1
model.add(Dense(100,activation='relu',input_dim=11,kernel_regularizer=l2(0.0001)))
model.add(Dropout(0.05, noise_shape=None, seed=None))
# Hidden Layer-2
model.add(Dense(100,activation='relu',kernel_regularizer=l2(0.0001)))
model.add(Dropout(0.05, noise_shape=None, seed=None))
# Output layer
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model_output = model.fit(X_train,y_train,epochs=100,batch_size=1000,verbose=1,validation_data=(X_valid,y_valid),)

print("\nFinal Training Accuracy", model_output.history['acc'][-1],
      "\nFinal Training Loss", model_output.history['loss'][-1],
      "\nFinal Validation Accuracy", model_output.history['val_acc'][-1],
      "\nFinal Validation Loss", model_output.history['val_loss'][-1])

print('\nMean Training Accuracy : ' , np.mean(model_output.history["acc"]))
print('Mean Validation Accuracy : ' , np.mean(model_output.history["val_acc"]))

# Plot training & validation accuracy values
plt.plot(model_output.history['acc'])
plt.plot(model_output.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show();

# Plot training & validation loss values
plt.plot(model_output.history['loss'])
plt.plot(model_output.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show();

# Test set classification report 
y_pred = model.predict(X_test,batch_size=1000, verbose=1)
y_pred[y_pred <= 0.5] = 0.
y_pred[y_pred > 0.5] = 1.

print("\n",classification_report(y_test, y_pred))

# Loading test.txt dataset and pre processing
data = pd.read_csv('test.txt', sep=' ', header=None) # pandas used as it automatically parses scientific notation
X_data = data.loc[:,1:].values # getting as array
# Performing Contrastive Principal Componenet Analysis
CPCA_data = mdl.fit_transform(X_data, classification_One[:len(X_data),:], alpha_selection='manual', alpha_value=12)
CPCA_data[:,0] = np.negative(CPCA_data[:,0]) # inverted X axis as data loaded inverted

# calculate kernel denisty estimations(takes a while to complete)
if run_kde == True:
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(CPCA_data)
    result = kde.score_samples(CPCA_data)
    np.savetxt('classDensity_data.txt', result)

classDensity_data = np.loadtxt('classDensity_data.txt')

# Assembling final feature set 
testData_ = np.zeros((len(CPCA_data), n_features+9))
testData_[:,:n_features] = CPCA_data # cPCA 2D reduced feature set
testData_[:,n_features:n_features+8] = X_data # original 8 dimensonal feature set
testData_[:,n_features+8] = classDensity_data # Results of KDE
X_data = testData_

data_y_pred = model.predict(X_data,batch_size=1000, verbose=1) # Predicting with model 
np.savetxt('test-labels.txt', data_y_pred)

data_y_pred = data_y_pred.ravel() # flatten for plotting
# Plotting test.txt data and coloring by predicted label
import matplotlib.pyplot as plt
fig1, ax1 = plt.subplots(1, 1)
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(CPCA_data[:,0], CPCA_data[:,1], c=data_y_pred, cmap=plt.cool(),s=0.4)
plt.show();