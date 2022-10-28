
from random import sample
import numpy as np
import pandas as pd 
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split#train_test_split để chia dữ liệu thành cặp train/test thường là 70/30
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import precision_score#Độ chính xác là tỷ số trong đó số lượng dương tính thật và số lượng dương tính giả
from sklearn.metrics import recall_score# tỷ lệ trong đó có số lượng âm tính thực sự và số lượng âm tính giả
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

df = pd.read_csv('./nam.csv') #đọc file csv
le = preprocessing.LabelEncoder()# thu vien labelendcoder de chuyen du lieu


df.loc[len(df.index)] = ['p','x','s','y','t','a','f','c','n','k','e','e','s','s','w','w','p','w','o','p','k','s','u']# gán dữ liệu vào cuối 
data = df.apply(le.fit_transform)# chuyen doi du lieu string to float
datatrain = data.iloc[-1:,1:]
print(data)
print(datatrain)
df = df.drop(np.shape(df)[0]-1)
data = data.drop(np.shape(data)[0]-1)


# print(data)
# print(df.iloc[-1:,1:])
X_data = data.iloc[:,1:]
y_data = data.iloc[:,0]
max = 0
for i in range (1,np.shape(X_data)[1]+1):
    pca = PCA(n_components=i).fit(X_data)
    X_new =pca.transform(X_data)
    X_train,X_test,y_train,y_test = train_test_split(X_new,y_data, test_size=0.3, shuffle = False)#shuffle đảo lộn dữ liệu test_size là số % dữ liệu test
    clfID3 = tree.DecisionTreeClassifier(criterion="entropy")
    clfID3.fit(X_train,y_train)
    y_predictID3 = clfID3.predict(X_test)
    print("ti le",i,"=>>",accuracy_score(y_test,y_predictID3))
    if(accuracy_score(y_test,y_predictID3)>max):
        max = accuracy_score(y_test,y_predictID3)
        Ncomponents=i
        pca_best = pca
        print("gan gia tri max",max)
        modelMax = clfID3
sample_pca = pca_best.transform(datatrain)
y_predict = modelMax.predict(sample_pca)

if(y_predict==0):
    {print("Nhãn của dữ liệu nhập vào là ",y_predict,"==> Không có độc")}    
else:
    {print("Nhãn của dữ liệu nhập vào là ",y_predict,"==> Có độc")}    




#perceptron()
pla = Perceptron()
pla.fit(X_train, y_train)
y_predictpla = pla.predict(X_test)

#ID3()
clfID3 = tree.DecisionTreeClassifier(criterion="entropy")
clfID3.fit(X_train,y_train)
y_predictID3 = clfID3.predict(X_test)

#Cart()
clfCart = tree.DecisionTreeClassifier(criterion='gini')#entropy 
clfCart.fit(X_train,y_train)
y_predictCart = clfCart.predict(X_test)



#ham tinh ti le du doan presesion,recall,f1 su dungj thuat toan
def  ThuatToantinhDoDo(y_dudoan):
    TP =0#Dương tính thực,Số lượng dữ liệu thuộc lớp ci(0,1) được phân loại chính xác vào lớp ci(0,1)
    TN =0#Đúng phủ định Sốlượng dữ liệu không thuộclớp ci được phân loại (chínhxác)
    FP=0#Dương tính giả Số lượng dữ liệu bên ngoài bị phân loại nhầm vào lớp ci
    FN =0#Phủ định sai.Sốlượng dữ liệu thuộc lớp ci bịphân loại nhầm (vào cáclớp khác ci)
    print("chieu dai :",len(y_dudoan))
    y_thucte = np.array(y_test)
    for i in range(0,len(y_dudoan)):
        if(y_thucte[i] == y_dudoan[i]):
            if(y_thucte[i]==1):
                TP = TP +1
            else:
                TN = TN +1
        else:
            if(y_thucte[i]==1):
                FN = FN +1
            else:
                FP = FP +1     
    print(TP,TN,FP,FN)

    print('Du doan Accuracy: ', (TP+TN)/len(y_dudoan))
    # du doan tong quat cho toan bo cac lop C={0,1}
    print("Du doan precision trung binh vi mo :",(TP+TN)/len(y_dudoan))# (TP+TN)/len(y_dudoan)
    print("Du doan precision trung binh vix mo :",(TP/(TP+FP)+TN/(TN+FN))/2)#x=(TP/(TP+FP)+TN/(TN+FN))/2
    
    # du doan theo lop 1 
    # print("Du doan precision voi lop C=1 :",(TP)/(TP+FP))#y du doan dung = 1/y du doan =1 khi dự đoán cây nấm có độc hay k , tỷ lệ này chính xác là ....
    # du doan tong quat cho toan bo cac lop C={0,1}
    print("Du doan recall trung binh vi mo :",(TP+TN)/len(y_dudoan))# (TP+TN)/len(y_dudoan)
    print("Du doan recall trung binh vix mo :",(TP/(TP+FN)+TN/(TN+FP))/2)#y=(TP/(TP+FN)+TN/(TN+FP))/2
    # du doan theo lop 1 
    # print("Du doan recall voi lop C=1 :",(TP)/(TP+FN))#y du doan dung = 1/y thuc te =1 ti le cây nấm có độc thuc te la
    # print("Du doan recall voi lop C=0 :",(TN)/(TN+FP))#y du doan dung = 1/y thuc te =1 ti le cây nấm có độc thuc te la

    # du doan tong quat cho toan bo cac lop C={0,1}
    print("Du doan F1_score trung binh vi mo :",2*((TP+TN)/len(y_dudoan)*(TP+TN)/len(y_dudoan))/((TP+TN)/len(y_dudoan)+(TP+TN)/len(y_dudoan)))#
    x=((TP/(TP+FP)+TN/(TN+FN))/2.0)
    y=((TP/(TP+FN)+TN/(TN+FP))/2.0)
    print("Du doan F1_score trung binh vix mo :",2.0/((1.0/x)+(1.0/y)))#2(x*y)/(x+y)2/(1/x+1/y)
    # du doan theo lop 1 
    # F1=2 * (((TP)/(TP+FP)) * ((TP)/(TP+FN))) / (((TP)/(TP+FP)) + ((TP)/(TP+FN)))
    # print("Du doan f1_score voi lop C=1 :",F1)#2 * (((TP)/(TP+FP)) * ((TP)/(TP+FN))) / (((TP)/(TP+FP)) + ((TP)/(TP+FN)))
#ham tinh ti le du doan presesion,recall,f1 su dungj thu vien sklearn
def  SklearnTinhDoDo(y_dudoan):           
    print('Du doan Accuracy: ', accuracy_score(y_test,y_dudoan))
    # du doan tong quat cho toan bo cac lop C={0,1}
    print("Du doan precision trung binh vi mo :",precision_score(y_test,y_dudoan,average="micro"))# (TP+TN)/len(y_dudoan)
    print("Du doan precision trung binh vix mo :",precision_score(y_test,y_dudoan,average="macro"))#x=(TP/(TP+FP)+TN/(TN+FN))/2

    print("Du doan recall trung binh vi mo :",recall_score(y_test,y_dudoan,average="micro"))# (TP+TN)/len(y_dudoan)
    print("Du doan recall trung binh vix mo :",recall_score(y_test,y_dudoan,average="macro"))#y=(TP/(TP+FN)+TN/(TN+FP))/2
    # du doan tong quat cho toan bo cac lop C={0,1}
    print("Du doan F1_score trung binh vi mo :",f1_score(y_test,y_dudoan,average="micro"))#
    print("Du doan F1_score trung binh vix mo :",f1_score(y_test,y_dudoan,average="macro"))#2(x*y)/(x+y)

# # su dung ham
# #perceptron()
# print("Phân tích kết quả của chương trình chạy thuật toán Perceptron")
# SklearnTinhDoDo(y_predictpla)

# print("==========================================================")
# # #ID3
# print("Phân tích kết quả của chương trình chạy thuật toán ID3")
# SklearnTinhDoDo(y_predictID3)
# print("==========================================================")
# #Cart
# print("Phân tích kết quả của chương trình chạy thuật toán Cart")
# SklearnTinhDoDo(y_predictCart)
# print("==========================================================")

