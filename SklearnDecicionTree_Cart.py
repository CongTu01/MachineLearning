
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



df = pd.read_csv('./weather.csv') #đọc file csv
data = df.apply(le.fit_transform)
#dt_train, dt_test = train_test_split(df, test_size=0.3, shuffle = False)#shuffle đảo lộn dữ liệu test_size là số % dữ liệu test

X_train = data.iloc[:,1:5]#iloc là truy vấn dữ liệu trong pandas chỉ int (loc: chấp nhận tham số đầu bảng(string))
y_train = data.iloc[:,5]


clf = tree.DecisionTreeClassifier(criterion='gini')#entropy
clf.fit(X_train,y_train)
pred = clf.predict(X_train)
print(pred)

