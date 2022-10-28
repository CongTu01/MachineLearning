from statistics import mode
import pandas as pd 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split#train_test_split để chia dữ liệu thành cặp train/test thường là 70/30
from sklearn import preprocessing
from sklearn import tree
le = preprocessing.LabelEncoder()# thu vien labelendcoder de chuyen du lieu

df = pd.read_csv('./weather.csv') #đọc file csv
data = df.apply(le.fit_transform)# chuyen doi du lieu string to float
dt_train, dt_test = train_test_split(data, test_size=0.3, shuffle = False)#shuffle đảo lộn dữ liệu test_size là số % dữ liệu test


X_train = dt_train.iloc[:,1:5]#iloc là truy vấn dữ liệu trong pandas chỉ int (loc: chấp nhận tham số đầu bảng(string))
y_train = dt_train.iloc[:,5]
X_test = dt_test.iloc[:,1:5]
y_test = dt_test.iloc[:,5]

model = SVC().fit(X_train,y_train)

print(model.predict(X_test))

