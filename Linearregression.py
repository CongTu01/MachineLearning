from sklearn import linear_model
import numpy as np

regr = linear_model.LinearRegression()# su dung ham hoi quy tuyen tinh
#gia nha , moi mau co 3 du lieu (dien tich, so phong ngu,cach trung tam)
X = np.array([[60,2,10],[40,2,5],[100,3,7]])
# gia nha thuc te theo tung mau
y = np.array([10,12,20])
regr.fit(X,y)# tinh w
x = np.array([[50,2,8]])
#gianhadudoan = regr.predict(np.array(x)# nha cang xa tt thi gia cang giam
gianhadudoan = x[0,0]*regr.coef_[0] + x[0,1]*regr.coef_[1] + x[0,2]*regr.coef_[2]+regr.intercept_# nha cang xa tt thi gia cang giam
print("gia nha du doan voi cap du lieu dau vao ",x, " la : ",gianhadudoan )

