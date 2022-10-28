from tkinter import *
from tkinter import messagebox
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split#train_test_split để chia dữ liệu thành cặp train/test thường là 70/30
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import precision_score#Độ chính xác là tỷ số trong đó số lượng dương tính thật và số lượng dương tính giả
from sklearn.metrics import recall_score# tỷ lệ trong đó có số lượng âm tính thực sự và số lượng âm tính giả
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

le = preprocessing.LabelEncoder()# thu vien labelendcoder de chuyen du lieu

df = pd.read_csv('./nam.csv') #đọc file csv
data = df.apply(le.fit_transform)# chuyen doi du lieu string to float
dt_train, dt_test = train_test_split(data, test_size=0.3, shuffle = False)#shuffle đảo lộn dữ liệu test_size là số % dữ liệu test


X_train_first = dt_train.iloc[:,1:]#iloc là truy vấn dữ liệu trong pandas chỉ int (loc: chấp nhận tham số đầu bảng(string))
y_train_first = dt_train.iloc[:,0]
X_test_first= dt_test.iloc[:,1:]
y_test_first = dt_test.iloc[:,0]

#form
form = Tk()
form.title("Dự đoán ,phân loại chất lượng của cây Nấm : ")
form.geometry("2400x1200")
form.configure(bg='pink')


lable_ten = Label(form, text = "Nhập thông tin cho cây nấm :", font=("Arial Bold", 13), fg="red")
lable_ten.grid(row = 1, column = 1, padx = 40)

lable_buying = Label(form, text = " Hình nắp: chuông = b, hình nón = c, lồi = x, phẳng = f, núm = k, trũng = s :")
lable_buying.grid(row = 2, column = 1)
textbox_1 = Entry(form)
textbox_1.grid(row = 3, column = 1)

lable_maint = Label(form, text = "Bề mặt nắp: sợi = f, rãnh = g, vảy = y, nhẵn = s")
lable_maint.grid(row = 4, column = 1)
textbox_2 = Entry(form)
textbox_2.grid(row = 5, column = 1)

lable_doors = Label(form, text = "Màu nắp: nâu = n, da bò = b, quế = c, xám = g, xanh lá cây = r, hồng = p, tím = u, đỏ = e, trắng = w, vàng = y")
lable_doors.grid(row = 6, column = 1)
textbox_3 = Entry(form)
textbox_3.grid(row = 7, column = 1)

lable_persons = Label(form, text = "Vết thâm: vết thâm = t, không = f")
lable_persons.grid(row = 8, column = 1)
textbox_4 = Entry(form)
textbox_4.grid(row = 9, column = 1)

lable_persons = Label(form, text = "Mùi: hạnh nhân = a, hồi = l, creosote = c, tanh = y, hôi = f, mốc = m, không = n, hăng = p, cay = s")
lable_persons.grid(row = 10, column = 1)
textbox_5 = Entry(form)
textbox_5.grid(row = 11, column = 1)

lable_persons = Label(form, text = "Mang-đính kèm: đính kèm = a, giảm dần = d, tự do = f, khía = n")
lable_persons.grid(row = 12, column = 1)
textbox_6 = Entry(form)
textbox_6.grid(row = 13, column = 1)

lable_persons = Label(form, text = "Khoảng cách mang: gần = c, đông đúc = w, xa = d")
lable_persons.grid(row = 14, column = 1)
textbox_7 = Entry(form)
textbox_7.grid(row = 15, column = 1)

lable_lug_boot = Label(form, text = "Kích thước mang: rộng = b, hẹp = n")
lable_lug_boot.grid(row = 16, column = 1 )
textbox_8 = Entry(form)
textbox_8.grid(row = 17, column = 1)

lable_safety = Label(form, text = "Màu mang: đen = k, nâu = n, buff = b, sô cô la = h, xám = g, xanh lá = r, cam = o, hồng = p, tím = u, đỏ = e, trắng = w, vàng = y")
lable_safety.grid(row = 18, column = 1)
textbox_9 = Entry(form)
textbox_9.grid(row = 19, column = 1)

lable_lug_boot = Label(form, text = "Hình dạng cuống: to ra = e, thon nhỏ = t")
lable_lug_boot.grid(row = 20, column = 1 )
textbox_10 = Entry(form)
textbox_10.grid(row = 21, column = 1)

lable_lug_boot = Label(form, text = "Gốc-cuống: củ = b, chùy = c, cốc = u, bằng = e, thân rễ = z, gốc = r, thiếu =?")
lable_lug_boot.grid(row = 22, column = 1 )
textbox_11 = Entry(form)
textbox_11.grid(row = 23, column = 1)

lable_ten = Label(form, text = "Nhập thông tin cho cây nấm :", font=("Arial Bold", 13), fg="red")
lable_ten.grid(row = 1, column = 3, padx = 40)

lable_lug_boot = Label(form, text = "Cuống-bề mặt-trên-vòng: sợi = f, vảy = y, mượt = k, mịn = s")
lable_lug_boot.grid(row = 2, column = 3 )
textbox_12 = Entry(form)
textbox_12.grid(row = 3, column = 3)

lable_lug_boot = Label(form, text = "Cuống-bề mặt-bên dưới-vòng: sợi = f, vảy = y, mượt = k, mịn = s")
lable_lug_boot.grid(row = 4, column = 3 )
textbox_13 = Entry(form)
textbox_13.grid(row = 5, column = 3)

lable_lug_boot = Label(form, text = "Cuống-màu-trên-vòng: nâu = n, buff = b, quế = c, xám = g, cam = o, hồng = p, đỏ = e, trắng = w, vàng = y")
lable_lug_boot.grid(row = 6, column = 3 )
textbox_14 = Entry(form)
textbox_14.grid(row = 7, column = 3)

lable_lug_boot = Label(form, text = "Cuống-màu-dưới-vòng: nâu = n, buff = b, quế = c, xám = g, cam = o, hồng = p, đỏ = e, trắng = w, vàng = y")
lable_lug_boot.grid(row = 8, column = 3 )
textbox_15 = Entry(form)
textbox_15.grid(row = 9, column = 3)

lable_lug_boot = Label(form, text = "Loại màn che: một phần = p, phổ quát = u")
lable_lug_boot.grid(row = 10, column = 3 )
textbox_16 = Entry(form)
textbox_16.grid(row = 11, column = 3)

lable_lug_boot = Label(form, text = "Màu màn che: nâu = n, cam = o, trắng = w, vàng = y")
lable_lug_boot.grid(row = 12, column = 3 )
textbox_17 = Entry(form)
textbox_17.grid(row = 13, column = 3)

lable_lug_boot = Label(form, text = "Số vòng: none = n, một = o, hai = t")
lable_lug_boot.grid(row = 14, column = 3 )
textbox_18 = Entry(form)
textbox_18.grid(row = 15, column = 3)

lable_lug_boot = Label(form, text = "Kiểu vòng:c, e, loe = f, lớn = l, không = n, mặt dây chuyền = p, vỏ bọc = s, vùng = z")
lable_lug_boot.grid(row = 16, column = 3 )
textbox_19 = Entry(form)
textbox_19.grid(row = 17, column = 3)

lable_lug_boot = Label(form, text = "Màu in bào tử: đen = k, nâu = n, buff = b, sô cô la = h, xanh lá cây = r, cam = o, tím = u, trắng = w, vàng = y")
lable_lug_boot.grid(row = 18, column = 3 )
textbox_20 = Entry(form)
textbox_20.grid(row = 19, column = 3)

lable_lug_boot = Label(form, text = "Dân số: nhiều = a, nhóm = c, nhiều = n, rải rác = s, nhiều = v, đơn độc = y")
lable_lug_boot.grid(row = 20, column = 3 )
textbox_21 = Entry(form)
textbox_21.grid(row = 21, column = 3)

lable_lug_boot = Label(form, text = "Môi trường sống: cỏ = g, lá = l, đồng cỏ = m, lối đi = p, thành thị = u, chất thải = w, rừng = d")
lable_lug_boot.grid(row = 22, column = 3 )
textbox_22 = Entry(form)
textbox_22.grid(row = 23, column = 3)

lable_ten = Label(form)
lable_ten.grid(row = 24, column = 1)
lable_ten = Label(form)
lable_ten.grid(row = 25, column = 1)
lable_ten = Label(form, text = "KẾT QUẢ TỈ LỆ DỰ ĐOÁN :", font=("Arial Bold", 15), fg="Blue")
lable_ten.grid(row = 27, column = 1)



#ID3()
treeID3 = tree.DecisionTreeClassifier(criterion="entropy")
treeID3.fit(X_train_first,y_train_first)


#Cart()
treeCart = tree.DecisionTreeClassifier(criterion='gini')#entropy 
treeCart.fit(X_train_first,y_train_first)


#cart
#dudoancarttheotest
y_predictCart = treeCart.predict(X_test_first)
lbl1 = Label(form)
lbl1.grid(column=1, row=30)
lbl1.configure(text="Tỉ lệ dự đoán đúng khi dùng CART : "+str(accuracy_score(y_test_first, y_predictCart)*100)+"%"+'\n'
                           +"Precision: "+str(precision_score(y_test_first, y_predictCart)*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test_first, y_predictCart)*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test_first, y_predictCart)*100)+"%"+'\n')

#ID3
#dudoanid3test
y_predictID3 = treeID3.predict(X_test_first)
lbl3 = Label(form)
lbl3.grid(column=2, row=30)
lbl3.configure(text="Tỉ lệ dự đoán đúng khi dùng ID3 : "+str(accuracy_score(y_test_first, y_predictID3)*100)+"%"+'\n'
                           +"Precision: "+str(precision_score(y_test_first, y_predictID3)*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test_first, y_predictID3)*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test_first, y_predictID3)*100)+"%"+'\n')
def cleardata():
    textbox_1.delete(0,END)
    textbox_2.delete(0,END)
    textbox_3.delete(0,END)
    textbox_4.delete(0,END)
    textbox_5.delete(0,END)
    textbox_6.delete(0,END)
    textbox_7.delete(0,END)
    textbox_8.delete(0,END)
    textbox_9.delete(0,END)
    textbox_10.delete(0,END)
    textbox_11.delete(0,END)
    textbox_12.delete(0,END)
    textbox_13.delete(0,END)
    textbox_14.delete(0,END)
    textbox_15.delete(0,END)
    textbox_16.delete(0,END)
    textbox_17.delete(0,END)
    textbox_18.delete(0,END)
    textbox_19.delete(0,END)
    textbox_20.delete(0,END)
    textbox_21.delete(0,END)
    textbox_22.delete(0,END)
def gandulieu():
    cleardata()
    textbox_1.insert(0,'x')
    textbox_2.insert(0,'s')
    textbox_3.insert(0,'y')
    textbox_4.insert(0,'t')
    textbox_5.insert(0,'a')
    textbox_6.insert(0,'f')
    textbox_7.insert(0,'c')
    textbox_8.insert(0,'b')
    textbox_9.insert(0,'k')
    textbox_10.insert(0,'e')
    textbox_11.insert(0,'c')
    textbox_12.insert(0,'s')
    textbox_13.insert(0,'s')
    textbox_14.insert(0,'w')
    textbox_15.insert(0,'w')
    textbox_16.insert(0,'p')
    textbox_17.insert(0,'w')
    textbox_18.insert(0,'o')
    textbox_19.insert(0,'p')
    textbox_20.insert(0,'n')
    textbox_21.insert(0,'n')
    textbox_22.insert(0,'g')
def gandulieucodoc():
    cleardata()#x,s,n,t,p,f,c,n,k,e,e,s,s,w,w,p,w,o,p,k,s,u
    textbox_1.insert(0,'x')
    textbox_2.insert(0,'s')
    textbox_3.insert(0,'n')
    textbox_4.insert(0,'t')
    textbox_5.insert(0,'p')
    textbox_6.insert(0,'f')
    textbox_7.insert(0,'c')
    textbox_8.insert(0,'n')
    textbox_9.insert(0,'k')
    textbox_10.insert(0,'e')
    textbox_11.insert(0,'e')
    textbox_12.insert(0,'s')
    textbox_13.insert(0,'s')
    textbox_14.insert(0,'w')
    textbox_15.insert(0,'w')
    textbox_16.insert(0,'p')
    textbox_17.insert(0,'w')
    textbox_18.insert(0,'o')
    textbox_19.insert(0,'p')
    textbox_20.insert(0,'k')
    textbox_21.insert(0,'s')
    textbox_22.insert(0,'u')
def getdata():
    data1 = textbox_1.get()
    data2 = textbox_2.get()
    data3 = textbox_3.get()
    data4 = textbox_4.get()
    data5 = textbox_5.get()
    data6 = textbox_6.get()
    data7 = textbox_7.get()
    data8 = textbox_8.get()
    data9 = textbox_9.get()
    data10 = textbox_10.get()
    data11 = textbox_11.get()
    data12= textbox_12.get()
    data13 = textbox_13.get()
    data14 = textbox_14.get()
    data15 = textbox_15.get()
    data16 = textbox_16.get()
    data17 = textbox_17.get()
    data18 = textbox_18.get()
    data19 = textbox_19.get()
    data20 = textbox_20.get()
    data21 = textbox_21.get()
    data22 = textbox_22.get()
    dataget = np.array(['p',data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16,data17,data18,data19,data20,data21,data22])
    return dataget
def checkdata():
    boolcheck = False
    check = getdata()
    for i in range(1,len(check)):
        if(check[i]==''):
            boolcheck=True
    return boolcheck
def dudoan():
    if(checkdata()):
        lblvalueID3.configure(text= '....')
        lblvalueCART.configure(text= '....')
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        df1 = pd.read_csv('./nam.csv') #đọc file csv
        check  = getdata()
        df1.loc[len(df1.index)] =check # gán dữ liệu vào cuối 
        data1 = df1.apply(le.fit_transform)# chuyen doi du lieu string to float
        datatrain = data1.iloc[-1:,1:]
        y_kquaID3 = treeID3.predict(datatrain)
        y_kquaCART=treeCart.predict(datatrain)
        if(y_kquaID3 == 1):
            lblvalueID3.configure(text= 'Cây nấm có độc')
        else:
            lblvalueID3.configure(text= 'Cây nấm KHÔNG có độc')
        if(y_kquaCART == 1):
            lblvalueCART.configure(text= 'Cây nấm có độc')
        else:
            lblvalueCART.configure(text= 'Cây nấm KHÔNG có độc')
        cleardata()
# Button
Lablevalue = Label(form, text = 'Kết quả dự đoán theo ID3 : ',font=("Arial Bold", 10), fg="Blue")
Lablevalue.grid(row = 3, column = 2, padx = 30)
lblvalueID3 = Label(form, text="...",bg='white')
lblvalueID3.grid(column=2, row=4)
Lablevalue = Label(form, text = 'Kết quả dự đoán theo CART :  ',font=("Arial Bold", 10), fg="Blue")
Lablevalue.grid(row = 6, column = 2, padx = 30)
lblvalueCART = Label(form, text="...",bg='white')
lblvalueCART.grid(column=2, row=7)
button_cart1 = Button(form, text = 'DỰ ĐOÁN', command = dudoan,fg="White",bg='blue')
button_cart1.grid(row = 10, column = 2, padx = 30)
button_cart1 = Button(form, text = 'set data', command = gandulieu)
button_cart1.grid(row = 29, column = 3, padx = 10)
button_cart1 = Button(form, text = 'set data', command = gandulieucodoc)
button_cart1.grid(row = 30, column = 3, padx = 10)

form.mainloop()