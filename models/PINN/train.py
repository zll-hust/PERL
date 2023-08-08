import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, Dropout, TimeDistributed, Activation, Flatten
from keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import data as dt
from custom_layers import IDM_Layer
from datetime import datetime


class ConvergenceChecker(Callback):
    def __init__(self, patience=3, threshold=0.0001):
        super(ConvergenceChecker, self).__init__()
        self.patience = patience
        self.threshold = threshold
        self.loss_history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss:
            self.loss_history.append(loss)
            if len(self.loss_history) > self.patience and abs(loss - self.loss_history[-self.patience - 1]) < self.threshold:
                print("Converged.")
                self.model.stop_training = True


def create_PINN_model(train_x_shape, train_y_shape):
    # IDM Layer
    vi_input = Input(shape=(1,), name='vi_input')  # 一个形状为(N, 1)的数组
    delta_v_input = Input(shape=(1,), name='delta_v_input')
    delta_d_input = Input(shape=(1,), name='delta_d_input')
    x_input = Input(shape=(None, 5), name='x_input')  # 一个形状为(N, T, 5)的数组，T是序列的长度。Assuming you've 6 features for LSTM
    # IDM Layer 输出
    idm_output = IDM_Layer(forward_steps=10)([vi_input, delta_v_input, delta_d_input])
    idm_output = Flatten()(idm_output)

    # LSTM Layer
    lstm_output = LSTM(256, return_sequences=True)(x_input)
    lstm_output = Dropout(0.2)(lstm_output)  # dropout层
    lstm_output = LSTM(256, return_sequences=False)(lstm_output)  # 返回整个序列
    lstm_output = Dropout(0.2)(lstm_output)  # dropout层
    lstm_output = Dense(10)(lstm_output)  # 将LSTM输出的每一个时间步通过一个Dense层进行处理
    lstm_output = Activation("relu")(lstm_output)

    final_output = 0.1*idm_output + 0.9*lstm_output
    pinn_model = Model(inputs=[vi_input, delta_v_input, delta_d_input, x_input], outputs=final_output)
    pinn_model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

    return pinn_model

# 主程序部分
with tf.device('/gpu:0'):
# if __name__ == '__main__':
    # 准备数据
    train_x, _, train_y, _, _, _ = dt.load_data()

    # 将train_x解构为IDM和LSTM的输入部分
    vi_data = np.array([item[0][0] for item in train_x])
    delta_v_data = np.array([item[0][1] for item in train_x])
    delta_d_data = np.array([item[0][2] for item in train_x])
    x_data = np.array([item[1] for item in train_x])
    backward = 50
    x_data = x_data.reshape((-1, backward, 5))
    train_y = np.array(train_y)

    # 创建模型
    PINN_model = create_PINN_model(x_data.shape, train_y.shape[1])

    # 添加收敛检查
    early_stopping = EarlyStopping(monitor='loss', patience=3, min_delta=0.0001, verbose=1)

    # 保存收敛过程
    history = PINN_model.fit([vi_data, delta_v_data, delta_d_data, x_data], train_y, epochs=50, batch_size=32, callbacks=[early_stopping])
    loss_history = history.history['loss']
    np.savetxt("./results/convergence_rate.csv", loss_history, delimiter=",")

    # 保存模型
    PINN_model.save("./model/NGSIM.h5")

    # 输出训练后的IDM参数
    idm_layer = [layer for layer in PINN_model.layers if isinstance(layer, IDM_Layer)][0]
    print("vf:", K.get_value(idm_layer.vf))
    print("A:", K.get_value(idm_layer.A))
    print("b:", K.get_value(idm_layer.b))
    print("s0:", K.get_value(idm_layer.s0))
    print("T:", K.get_value(idm_layer.T))

    with open("./results/IDM_parameter_train_results.txt", 'a') as f:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f'{current_time}\n')
        f.write(f'IDM parameters trained result vf, A, b, s0, T = '
                f'{K.get_value(idm_layer.vf)}'
                f'{K.get_value(idm_layer.A)}'
                f'{K.get_value(idm_layer.b)}'
                f'{K.get_value(idm_layer.s0)}'
                f'{K.get_value(idm_layer.T)}\n')


def save_results_to_file(filename, mse, mse2):
    with open(filename, 'a') as f:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f'{current_time}\n')
        f.write(f'MSE when predicting multi steps acceleration: {mse}\n')
        f.write(f'MSE when predicting first acceleration: {mse2}\n\n')