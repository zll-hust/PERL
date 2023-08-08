import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import argparse
import os
from datetime import datetime
import tensorflow as tf

import sys
sys.path.append('/home/ubuntu/Documents/PERL/models/')
from Data_driven_LSTM import data as dt

def save_results_to_file(filename, mse, mse2):
    with open(filename, 'a') as f:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f'{current_time}\n')
        f.write(f'MSE when predicting multi steps acceleration: {mse}\n')
        f.write(f'MSE when predicting first acceleration: {mse2}\n\n')


def plot_and_save_prediction(A_real, A_hat, sample_id):
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(A_real)), A_real, color='b', markersize=1, label='Real-world')
    plt.plot(np.arange(len(A_real)), A_hat, color='r', markersize=1, label='LSTM')
    plt.xlabel('Index')
    plt.ylabel('Acceleration error $(m/s^2)$')
    plt.ylim(-4, 4)
    plt.title(f'Sample ID: {sample_id}')
    plt.legend()
    plt.savefig(f'./results/plots/predict_result_{sample_id}.png')
    plt.close()


with tf.device('/gpu:0'):
    _, test_x, _, test_y_real, A_min, A_max = dt.load_data()
    model = load_model("./model/NGSIM.h5")
    test_y_predict = model.predict(test_x)

    A_real = test_y_real.tolist()
    A_hat  = test_y_predict.tolist()

    # 反归一化
    A_real = np.array(A_real) * (A_max - A_min) + A_min
    A_hat  = np.array(A_hat) * (A_max - A_min) + A_min

    mse = mean_squared_error(A_real, A_hat)
    print('MSE when predicting multi steps acceleration:', mse)
    mse2 = mean_squared_error(A_real[:,0], A_hat[:,0])
    print('MSE when predicting first acceleration:', mse2)

    save_results_to_file('./results/predict_MSE_results.txt', mse, mse2)

    os.makedirs('./results/plots', exist_ok=True)
    # for i in range(len(A_real)):
    #     plot_and_save_prediction(A_real[i], A_hat[i], i)