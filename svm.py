# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import svm

data = pd.read_csv(filepath_or_buffer = 'C:/Users/admin/Desktop/homework/6.2/watermelon3_0_En.csv')

#Seperata the data into samples and labels
columns = data.columns
attributes = columns[:-1]
label = columns[-1]
samples_frame = data[attributes]
label_series = data[label]
#Up to now, we do not need DataFrame any more, instead, an array is more suitable
samples = samples_frame.values
labels = label_series.tolist()

#linear kernel
clf = svm.SVC(kernel = 'linear')
clf.fit(samples,labels)
#Get support vectors
print(clf.support_vectors_)
#Get indices of support vectors
print(clf.support_)
#Get number of support vectors for each classes
print('linear kernel',clf.n_support_)

#gaussian kernel
clf1 = svm.SVC(kernel = 'rbf')
clf1.fit(samples,labels)
#Get support vectors
print(clf1.support_vectors_)
#Get indices of support vectors
print(clf1.support_)
#Get number of support vectors for each classes
print('gaussian kernel',clf1.n_support_)
