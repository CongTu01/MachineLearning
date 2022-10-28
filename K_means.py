import pandas as pd 
from sklearn.model_selection import train_test_split#train_test_split để chia dữ liệu thành cặp train/test thường là 70/30
from sklearn import preprocessing
from sklearn.cluster import KMeans
le = preprocessing.LabelEncoder()# thu vien labelendcoder de chuyen du lieu
from sklearn.metrics import f1_score

df = pd.read_csv('./weather.csv') #đọc file csv
data = df.apply(le.fit_transform)#chuyen doi du lieu string to float
dt_train, dt_test = train_test_split(data, test_size=0.3, shuffle = False)#shuffle đảo lộn dữ liệu test_size là số % dữ liệu test


X_train = dt_train.iloc[:,1:5]#iloc là truy vấn dữ liệu trong pandas chỉ int (loc: chấp nhận tham số đầu bảng(string))
y_train = dt_train.iloc[:,5]
X_test = dt_test.iloc[:,1:5]
y_test = dt_test.iloc[:,5]


kmean = KMeans(n_clusters=2,init="k-means++").fit(X_train)# phan thanh 2 cum //du lieu dau vao k duoc gan nhan
print("Vị trí center của từng cụm :")
print(kmean.cluster_centers_)#vi tri trung tam cua tung cum
print(kmean.labels_)
pre_kmean = kmean.predict(X_test)
print(" du doan phan cum với cặp dữ liệu :")
print(pre_kmean)


