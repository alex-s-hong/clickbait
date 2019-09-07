import numpy as np
import math
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, MaxPooling1D
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils

import matplotlib.pyplot as plt
%matplotlib inline

from sklearn import metrics
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier

from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

integrated_df =pd.read_csv('vector_form.csv')

#image representation clipping
image_col = ['person_y', 'bicycle', 'car_y', 'motorcycle', 'airplane', 'bus', 'train_y', 'truck_y', 'boat_y',
 'traffic light', 'fire hydrant',  'stop sign', 'parking meter', 'bench_y', 'bird_y', 'cat_y', 'dog_y', 'horse',
 'sheep_y', 'cow_y', 'elephant', 'bear_y', 'zebra_y', 'giraffe', 'backpack_y', 'umbrella_y', 'handbag_y', 'tie_y',
 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard_y',
 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup_y', 'fork_y', 'knife_y', 'spoon', 'bowl_y', 'banana_y', 'apple',
 'sandwich_y', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza_y', 'donut_y', 'cake_y', 'chair_y', 'couch', 'potted plant',
 'bed_y', 'dining table', 'toilet_y', 'tv_y', 'laptop_y', 'mouse', 'remote', 'keyboard_y', 'cell phone', 'microwave', 'oven_y',
 'toaster_y', 'sink_y', 'refrigerator', 'book_y', 'clock_y', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# normalizing the image representation
# clipping to 2 instead of 1 to reserve the multiple objects in image

for col_name in image_col:
  for i in range(len(integrated_df)):
    if integrated_df[col_name][i] >2:
      integrated_df.set_value(i, col_name, 2)
      
      
col= ['id_y', 'postMedia', 'postText', 'postTimestamp', 'targetCaptions', 'targetDescription',	
'targetKeywords',	'targetParagraphs',	'targetTitle',	'truthClass',	'truthJudgments',	
'truthMean',	'truthMedian',	'truthMode', 'truthClassInt']


truth = pd.DataFrame(integrated_df['truthClassInt'])

integrated_df = integrated_df.drop(col,axis =1)

integrated_df, X_test, truth, Y_test = train_test_split(integrated_df, truth, test_size = 0.2)

#create data frame only with text which will be used for comparision baseline
text_df = integrated_df.drop(image_col, axis = 1)
X_test_t = X_test.drop(image_col, axis = 1)

#make data frames into numpy arrays
inte_df_array = np.array(integrated_df)
text_df_array = np.array(text_df)
X_test_array = np.array(X_test)
X_test_t_array = np.array(X_test_t)
truth_array= np.array(truth)
Y_test_array = np.array(Y_test)

def cv_training(clf_list, X, Y, x_test, y_test):
  results = []
  names = []
  res_list = []
  scoring_method = 'accuracy'
  for name, model in clf_list:
    print('classifier:', name)
    kfold = KFold(n_splits=10, random_state = 5)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring_method)
    results.append(cv_results)
    names.append(name)
    print('1. Cross Validation Info')
    msg = "%s: mean= %f, std = %f" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
    #test on test data
    print("Testing")
    model.fit(X,Y)
    y_pred = model.predict(x_test)
    test_result = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix :')
    print(test_result)
    res = accuracy_score(y_test, y_pred)
    res_list.append(res)
    print('Accuracy Score:', res)
    print('Report:')
    print (classification_report(y_test,y_pred))
    print('AUC Score: ',metrics.roc_auc_score(y_test, y_pred))
   
  #boxplot algorithm comparision
  fig = plt.figure()
  fig.suptitle('Model Comparison')
  ax = fig.add_subplot(111)
  plt.boxplot(results)
  ax.set_xticklabels(names)
  plt.show()
  
  return res_list

clf_list = []
clf_list.append(('CART', DecisionTreeClassifier()))
clf_list.append(('NB', MultinomialNB()))
clf_list.append(('LR', LogisticRegression()))
#clf_list.append(('KNN', KNeighborsClassifier()))
clf_list.append(('GB', GradientBoostingClassifier()))
clf_list.append(('RF', RandomForestClassifier()))

baseline_result = cv_training(clf_list, text_df_array, truth_array, X_test_t_array, Y_test_array)
txt_img_result = cv_training(clf_list, inte_df_array, truth_array, X_test_array, Y_test_array)

print(baseline_result)
print(txt_img_result)

#CNN model

def CNN_model(X, Y, X_test, Y_test):
  
  # extract the valid sets
  sss = StratifiedShuffleSplit(test_size=0.1, random_state=10)
  for train_index, valid_index in sss.split(X, Y):
      X_train, X_valid = X[train_index], X[valid_index]
      y_train, y_valid = Y[train_index], Y[valid_index]
  
  print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
  
  nrows, ncols = X_train.shape
  valid_rows = len(X_valid)
  
  # reshape the array
  X_train = X_train.reshape(nrows, ncols, 1)
  X_valid = X_valid.reshape(valid_rows, ncols, 1)
  y_train = np_utils.to_categorical(y_train, 2)
  y_valid = np_utils.to_categorical(y_valid, 2)
  
  nrows_t, nrows_c = X_test.shape
  X_test = X_test.reshape(nrows_t, nrows_c, 1)
  #Y_test = np_utils.to_categorical(Y_test, 2)
  
  # convolutional neural network model
  
  model = Sequential()
  model.add(Convolution1D(filters=100, kernel_size=10, activation='relu', input_shape = (ncols, 1)))
  model.add(BatchNormalization())
  model.add(MaxPooling1D(3))
  model.add(Convolution1D(filters=100, kernel_size=10, activation='relu'))
  #model.add(BatchNormalization())
  model.add(MaxPooling1D(3))
  model.add(Flatten())
  model.add(Dropout(0.2))
  model.add(Dense(2, activation='softmax'))
  print(model.summary)
  
  sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
  model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
  
  nb_epoch = 30
  model.fit(X_train, y_train, epochs=nb_epoch, validation_data=(X_valid, y_valid), batch_size=120)
  

  return model
  
  
baseline_cnn = CNN_model(text_df_array, truth_array, X_test_t_array, Y_test_array)
inte_model = CNN_model(inte_df_array, truth_array, X_test_array, Y_test_array)

X_test_t_array = X_test_t_array.reshape(len(X_test_t_array), 13668, 1)
y_pred = baseline_cnn.predict(X_test_t_array)


for i in range (len(y_pred)):
  if y_pred[i][0] < y_pred[i][1]:
    y_pred[i][0], y_pred[i][1] = 0, 1
  else:
    y_pred[i][0], y_pred[i][1] = 1, 0
    
print(accuracy_score(Y_test_array, y_pred)

y_pred2 = inte_model.predict(X_test_array)

for i in range (len(y_pred2)):
  if y_pred[i][0] < y_pred2[i][1]:
    y_pred2[i][0], y_pred2[i][1] = 0, 1
  else:
    y_pred2[i][0], y_pred2[i][1] = 1, 0
    
accuracy_score(Y_test_array, y_pred2)







