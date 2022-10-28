
import pandas as pd 
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

le = preprocessing.LabelEncoder()

df = pd.read_csv('./nam.csv')
data = df.apply(le.fit_transform)
dt_train, dt_test = train_test_split(data, test_size=0.3, shuffle = False)


X_train = dt_train.iloc[:,1:]
y_train = dt_train.iloc[:,0]
X_test = dt_test.iloc[:,1:]
y_test = dt_test.iloc[:,0]


#perceptron(ct)
pla = Perceptron()
pla.fit(X_train, y_train)
y_predictpla = pla.predict(X_test)

#ID3(ct)
clfID3 = tree.DecisionTreeClassifier(criterion="entropy")
clfID3.fit(X_train,y_train)
y_predictID3 = clfID3.predict(X_test)

#Cart(ct)
clfCart = tree.DecisionTreeClassifier(criterion='gini')
clfCart.fit(X_train,y_train)
y_predictCart = clfCart.predict(X_test)


def  SklearnTinhDoDo(y_dudoan):           
    print('Du doan Accuracy: ', accuracy_score(y_test,y_dudoan))
    print("Du doan precision trung binh vix mo :",precision_score(y_test,y_dudoan,average="macro"))
    print("Du doan recall trung binh vix mo :",recall_score(y_test,y_dudoan,average="macro"))
    print("Du doan F1_score trung binh vix mo :",f1_score(y_test,y_dudoan,average="macro"))

print("Phân tích kết quả của chương trình chạy thuật toán Perceptron")
SklearnTinhDoDo(y_predictpla)

# #ID3
print("Phân tích kết quả của chương trình chạy thuật toán ID3")
SklearnTinhDoDo(y_predictID3)

#Cart
print("Phân tích kết quả của chương trình chạy thuật toán Cart")
SklearnTinhDoDo(y_predictCart)

