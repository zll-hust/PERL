from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
import argparse
from datetime import datetime
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
sys.path.append('/home/ubuntu/Documents/PERL/models/')
from IDM_NN import data as dt

def save_results_to_file(filename, mse, mse2):
    with open(filename, 'a') as f:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f'{current_time}\n')
        f.write(f'MSE when predicting multi steps acceleration: {mse}\n')
        f.write(f'MSE when predicting first acceleration: {mse2}\n\n')


if __name__ == '__main__':
    forward = 10
    backward = 50
    # 准备数据
    _, test_x, _, test_rows, a_residual_IDM_min, a_residual_IDM_max, test_chain_ids = dt.load_data()
    test_x = test_x.reshape(test_x.shape[0], backward, 6)

    # 加载模型
    model = load_model("./model/NGSIM.h5")

    # 在测试集上进行预测
    A_error_hat = model.predict(test_x)

    # 反归一化
    A_error_hat = A_error_hat * (a_residual_IDM_max - a_residual_IDM_min) + a_residual_IDM_min

    # 找到原始数据作为对比
    df = pd.read_csv("/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_IDM_results.csv")
    indices = []
    for chain_id in test_chain_ids:
        chain_df = df[df['chain_id'] == chain_id]
        indices.extend(chain_df.index[-forward:])
    # 使用这些索引从A_IDM中提取数据
    A_IDM_array = df['a_IDM'].iloc[indices].to_numpy()
    n_samples = len(A_IDM_array) // forward
    A_IDM = A_IDM_array.reshape(n_samples, forward)

    A_array = df['a'].iloc[indices].to_numpy()
    A = A_array.reshape(n_samples, forward)

    # 计算A_PERL
    A_PERL = A_IDM - A_error_hat

    # 计算MSE
    mse = mean_squared_error(A, A_PERL)
    print('MSE when predicting all acceleration:', mse)
    mse2 = mean_squared_error(A[:, 0], A_PERL[:, 0])
    print('MSE when predicting first acceleration:', mse2)
    save_results_to_file('./results/predict_MSE_results.txt', mse, mse2)

    # 绘制预测结果和真实值的图形
    os.makedirs('./results/plots', exist_ok=True)
    for i in range(n_samples):
        plt.figure(figsize=(6, 4))
        x = range(len(A_PERL[i]))
        plt.plot(x, A[i, :], color='b', markersize=0.5, label='Real-world')
        plt.plot(x, A_IDM[i, :], color='g', markersize=0.2, label='IDM')
        plt.plot(x, A_PERL[i, :], color='r', markersize=0.5, label='PERL(IDM+LSTM)')
        plt.xlabel('Time Index (0.1 s)')
        plt.ylabel('Acceleration $(m/s^2)$')
        plt.ylim(-4, 4)
        plt.legend()
        plt.savefig(f'./results/plots/PINN(IDM+NN)_result{i}.png')
        # plt.show()
        plt.close()









    # parser = argparse.ArgumentParser()
    # parser.add_argument('--platoon_num', type=int, help='Platoon number')
    # args = parser.parse_args()
    # platoon_num = args.platoon_num
    # #platoon_num = 20
    #
    # # 准备数据
    # _, test_x, _, test_y, test_rows, A_error_min, A_error_max, _,_ = dt.prepare_data(look_back = 50, look_forward = 30)
    #
    # # 加载模型
    # #model = load_model(f"./model/platoon{platoon_num}.h5")
    # model = load_model("./model/NGSim.h5")
    #
    # # 在测试集上进行预测
    # A_error_hat = model.predict(test_x)
    #
    # # 反归一化
    # A_error_hat = A_error_hat * (A_error_max - A_error_min) + A_error_min
    # A_error_hat = A_error_hat.flatten()
    #
    # # 获取对应行的A_hat值
    # df = pd.read_csv(f'../Data/ASta_platoon{platoon_num}_new1.csv')
    # A_hat = df.loc[test_rows, 'A_hat'].values
    # A2 = df.loc[test_rows, 'A2'].values
    #
    # # 计算A_pinn
    # A_pinn = A_hat - A_error_hat
    #
    # # 计算MSE
    # mse = mean_squared_error(A2, A_pinn)
    # print('MSE when predicting acceleration:', mse)
    # mse2 = mean_squared_error(A2[:][0], A_pinn[:][0])
    # print('MSE when predicting first acceleration:', mse2)
    #
    # plt.figure(figsize=(10, 4))
    # plt.plot(test_rows, A2, color='b', markersize=1, label='Original a')
    # plt.plot(test_rows, A_hat, color='g', markersize=0.5, label='IDM predict a')
    # plt.plot(test_rows, A_pinn, color='r', markersize=1, label='PINN Predicted a')
    # plt.xlabel('index')
    # plt.ylabel('Acceleration error (m/s^2)')
    # plt.ylim(-2, 2)
    # plt.title('MSE: {:.4f}'.format(mse))
    # plt.legend()
    # plt.savefig('../Results/Platoon{}_PINN_result.png'.format(platoon_num))
    # plt.show()
    #
