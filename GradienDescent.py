# Tìm điểm cực trị cục bộ của hàm f(x) = x**4 + 5*sin(x)**2

import numpy as np
#grad(x): Tính đạo hàm của hàm f(x)

x =np.array([[1,2],[-2,5],[0,1]])
y = np.array([[1],[6],[1]])
w = np.array([[1],[2],[3]])

def grad(x,y,w1,w2,w3):
    w = (np.array([[w1,w2,w3]])).transpose()
    z = w.dot(x[0][0])+w.dot(x[0][1])+w.dot(x[1][0])+w.dot(x[1][1])+w.dot(x[2][0])+w.dot(x[2][1])
    return (y-z)


#cost(x): Tính giá trị của hàm f(x)
def cost(x,y,w1,w2,w3):
    w = (np.array([[w1,w2,w3]])).transpose()
    z = w.dot(x[0][0])+w.dot(x[0][1])+w.dot(x[1][0])+w.dot(x[1][1])+w.dot(x[2][0])+w.dot(x[2][1])
    
    return (1/2)*((y-z)*(y-z))

#myGD1(eta, x0): Tìm điểm cực trị cục bộ của hàm f(x) theo công thức Gradient Descent: x(t+1) = x(t) - eta*grad(x)
#eta là learning rate (tốc độ học), x0 là điểm khởi tạo

def myGD1(x_X,y,eta, w1,w2,w3,k):
    w_new = (np.array([[w1],[w2],[w3]]))
    i=0
    while( (w_new>1e-3).all() and (i<k)):
        w_new = w_new - eta*grad(x_X,y,w_new[0][0],w_new[1][0],w_new[2][0])
        print(w_new)
        i=i+1
    # for it in range(100):
    #     x_new = x[-1] - eta*grad(x[-1])
    #     if abs(grad(x_new)) < 1e-3:
    #         break
    #     x.append(x_new)
    return w_new
# (x1, it1) = myGD1(.1, -5)
# (x2, it2) = myGD1(.1, 5)
print(myGD1(x,y,0.1,1,2,3,1000))
# print((grad(x,y,w[0][0],w[1][0],w[2][0])>1e-3).all())
# print(grad(x,y,w[0][0],w[1][0],w[2][0]))

#print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
#print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))