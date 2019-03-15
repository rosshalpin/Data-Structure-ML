# Training ML algorithm
Supervised machine learning

### Objective
Given a mixed, balanced and labeled training set of 100,000 inputs with 8 features, attempt to correctly label the given unlabeled test set of 10,000 inputs.

### Method
#### Pre-process training data, attempt to separate labeled data. 
First to separate the dataset, implemented Contrastive Principal Component Analysis (Abid et al, 2017),
which is "designed to discover low-dimensional structure that is unique to a dataset, or enriched in one dataset relative to other data", this "technique is a generalization of standard PCA". 

Reduced feature set to 2 dimensions using cPCA, contrasting the entire training set against a subset of the training set which is labeled as "1".  
![png](Graphs/Structure/output_2_0.png)  
The extra credit hidden data structure can be seen within the above scatter plot.  

I then calculated the Kernel Density Estimation(KDE) of these 2D points. The KDE algorithm, will smoothly fit the 2D points to a probability density function. From this smoothed distribution we get the estimated density values.

I then took the original 8 dimensional feature set, added the 2 dimensional reduced feature set and added the KDE values into an 11 dimensional feature set. 

This was split into training, test and validation sets.

#### Deep sequential neural network using Keras, TensorFlow.
```python
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

```

### Model Accuracy and Loss of training and validation sets over 100 epochs
![png](Graphs/Train/output_0_1.png) ![png](Graphs/Train/output_0_2.png)  
  
### Final model scores after 100 epochs. 
|label         | precision |   recall | f1-score  | support|
| ------------- |:-------------:| -----:| -----:| -----:|
|0.0   |    0.94  |    0.95  |    0.95  |    4872|
1.0    |   0.95    |  0.95   |   0.95  |    5128|

|         | precision |   recall | f1-score  | support|
| ------------- |:-------------:| -----:| -----:| -----:|
micro avg   |    0.95   |  0.95   |   0.95  |   10000|
macro avg   |    0.95   |   0.95   |   0.95  |   10000|
weighted avg   |    0.95  |    0.95   |   0.95  |   10000|

|Type         | Score | 
| ------------- |:-------------:|
|Final Training Accuracy |  0.9515999995172024|
|Final Training Loss |  0.1549298545345664|
|Final Validation Accuracy |  0.9485999941825867|
|Final Validation Loss |  0.16442587226629257|
|Mean Training Accuracy |  0.8881981247104704|  
|Mean Validation Accuracy |  0.8933929995894431|  
