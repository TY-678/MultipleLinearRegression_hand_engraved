import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


np.set_printoptions(formatter={'float': '{: .2e}'.format})

data_origin = pd.read_csv('tscs2020q1.csv')
data = data_origin.loc[:, ['v2y','v1', 'v76', 'v7a', 'v102']]
data.columns = ['age', 'sex', 'drink', 'education', 'work_hours']


#學歷國中以下0 高中以上1 大學2 碩士以上3
data['education'] = data['education'].map({1:0, 2:0, 3:0, 4:0, 5:0, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:1, 16:2, 17:2, 18:2, 19:2, 20:3, 21:3})


# 找到需要刪除的row
rows_to_drop = np.logical_or(data['age'] >= 997, data['work_hours'] >= 900)

# 更新 'age' col
data.loc[~rows_to_drop, 'age'] = 112 - data.loc[~rows_to_drop, 'age']

# 刪除需要刪除的row
data = data[~rows_to_drop]

#有遺漏值就整筆刪除
data.dropna(inplace=True)


x = data[['age', 'sex', 'drink', 'education']]
y = data['work_hours']

#train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)  #stratify=y


#StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


def compute_cost(x, y, w, b):
    y_pred = (w * x).sum(axis=1) + b
    cost = ((y - y_pred) ** 2).mean()
    return cost

def compute_gradient(x, y, w, b):
    y_pred = (w * x).sum(axis=1) + b
    b_gradient = (y_pred - y).mean()
    w_gradient = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        w_gradient[i] = (x[:, i] * (y_pred - y)).mean()
    
    return w_gradient, b_gradient

def gradient_descent(x, y, w_init, b_init, learning_rate, cost_function, gradient_function, run_iter, p_iter = 1000):
    w = w_init
    b = b_init
    w_hist = []
    b_hist = []
    c_hist = []
    for i in range(run_iter):
        w_gradient, b_gradient = gradient_function(x, y, w, b)
        w = w - w_gradient * learning_rate
        b = b - b_gradient * learning_rate
        cost = cost_function(x, y, w, b)
        w_hist.append(w)
        b_hist.append(b)
        c_hist.append(cost)
        if i % p_iter == 0:
            print(f'{i:6}, cost:{cost: .4e}, w:{w}, b:{b: .2e}, w_gradient:{w_gradient}, b_gradient:{b_gradient: .2e}')
    return w, b, w_hist, b_hist, c_hist



w = np.array([1, 2, 3, 4])
b = 0
run_iter = 10000
learning_rate = 0.003

w_final, b_final, w_hist, b_hist, c_hist = gradient_descent(x_train, y_train, w, b, learning_rate, compute_cost, compute_gradient, run_iter)
y_pred = (w_final * x_test).sum(axis=1) + b_final







# #test & error
# error = abs(y_test - y_pred)

# output = pd.DataFrame({'y_pred' : y_pred, 'y_test' : y_test, 'error' : error})
# print(output)






# plt.plot(range(len(y_pred))[::10], y_test[::10], label='Y_test')
# plt.plot(range(len(y_pred))[::10], y_pred[::10], label='Y_pred')
# plt.plot(range(len(y_pred))[::10], error[::10], label='error')
# plt.title('y_test vs y_pred  & error')
# plt.xlabel('number of objects')
# plt.ylabel('workhours')
# plt.legend()
# plt.show()


#w, b by cost
# fig = plt.figure()  # 創建一個新的圖表
# ax = fig.add_subplot(111, projection='3d')  # 創建一個3D坐標系

# w0 , w1, w2, w3 = [], [], [], []
# for i in range(len(w_hist)):
#     w0.append(w_hist[i][0])
#     w1.append(w_hist[i][1])
#     w2.append(w_hist[i][2])
#     w3.append(w_hist[i][3])
# ax.scatter(w0, range(run_iter), c_hist, marker='o', label='age')  # 繪製3D散點圖
# ax.scatter(w1, range(run_iter), c_hist, marker='o', label='sex')  # 繪製3D散點圖
# ax.scatter(w2, range(run_iter), c_hist, marker='o', label='drink')  # 繪製3D散點圖
# ax.scatter(w3, range(run_iter), c_hist, marker='o', label='education')  # 繪製3D散點圖
# ax.scatter(b_hist, range(run_iter), c_hist,  marker='o', label='b')  # 繪製3D散點圖


# ax.set_xlabel('w&b')  # 設置X軸標籤
# ax.set_ylabel('run_iter')  # 設置Y軸標籤
# ax.set_zlabel('cost')  # 設置Z軸標籤

# plt.title('W&B vs Cost')  # 設置圖表標題
# plt.legend()  # 顯示圖例
# plt.show()  # 顯示圖表


