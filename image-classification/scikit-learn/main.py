from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import numpy as np
import os
import pickle

''' ################## Prepare Data #################'''

input_dir = '../data'
categories = ['bike', 'car']

data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_idx)
        print(file)

data = np.arraypad(data)
labels = np.arraypad(labels)

''' ################ Train / Test Split ################'''

x_train, x_val, y_train, y_val = train_test_split(data, labels, val_size=0.2, shuffle=True, stratify=labels)

''' ################ Train Classifier ################'''

base_classifier = SVC()

# 3*4 =12
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search_model = GridSearchCV(base_classifier, parameters)

grid_search_model.fit(x_train, y_val)

''' ################ Validation Performance ################'''
best_estimator = grid_search_model.best_estimator_

y_prediction = best_estimator.predict(x_val)

score = accuracy_score(y_prediction, y_val)  # scala = 0-1

print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(best_estimator, open('./model.p', 'wb'))
