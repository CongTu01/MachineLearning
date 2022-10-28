
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import precision_score#Độ chính xác là tỷ số trong đó số lượng dương tính thật và số lượng dương tính giả
from sklearn.metrics import recall_score# tỷ lệ trong đó có số lượng âm tính thực sự và số lượng âm tính giả
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
le = preprocessing.LabelEncoder()# thu vien labelendcoder de chuyen du lieu
df = pd.read_csv('./nam.csv') #đọc file csv
data = df.apply(le.fit_transform)# chuyen doi du lieu string to float
dt_train, dt_test = train_test_split(data, test_size=0.3, shuffle = False)#shuffle đảo lộn dữ liệu test_size là số % dữ liệu test


X_train = dt_train.iloc[:,1:]#iloc là truy vấn dữ liệu trong pandas chỉ int (loc: chấp nhận tham số đầu bảng(string))
y_train = dt_train.iloc[:,0]
X_test = dt_test.iloc[:,1:]
y_test = dt_test.iloc[:,0]


#ID3()
clfID3 = tree.DecisionTreeClassifier(criterion="entropy")
clfID3.fit(X_train,y_train)
y_predictID3 = clfID3.predict(X_test)
print('Du doan Accuracy: ', accuracy_score(y_test,y_predictID3))
# du doan tong quat cho toan bo cac lop C={0,1}
print("Du doan precision trung binh vi mo :",precision_score(y_test,y_predictID3,average="micro"))# (TP+TN)/len(y_predictID3)
print("Du doan precision trung binh vix mo :",precision_score(y_test,y_predictID3,average="macro"))#x=(TP/(TP+FP)+TN/(TN+FN))/2
# du doan theo lop 1 
# print(precision_score(y_test,y_predictID3,pos_label=1))
# du doan tong quat cho toan bo cac lop C={0,1}
print("Du doan recall trung binh vi mo :",recall_score(y_test,y_predictID3,average="micro"))# (TP+TN)/len(y_predictID3)
print("Du doan recall trung binh vix mo :",recall_score(y_test,y_predictID3,average="macro"))#y=(TP/(TP+FN)+TN/(TN+FP))/2
# du doan theo lop 1 
# print(recall_score(y_test,y_predictID3,pos_label=1))
# print(recall_score(y_test,y_predictID3,pos_label=0))
# du doan tong quat cho toan bo cac lop C={0,1}
print("Du doan F1_score trung binh vi mo :",f1_score(y_test,y_predictID3,average="micro"))#
print("Du doan F1_score trung binh vix mo :",f1_score(y_test,y_predictID3,average="macro"))#2(x*y)/(x+y)
# du doan theo lop 1 
# print(f1_score(y_test,y_predictID3,pos_label=1))
