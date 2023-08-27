from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, BatchNormalization, Reshape
from keras.callbacks import EarlyStopping
import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import os
import data as dt

# import sys
# sys.path.append('/home/ubuntu/Documents/PERL/models/')
# from PERL_IDM_MLP import data as dt


def build_lstm_model_light(input_shape, forward):
    model = Sequential()
    model.add(LSTM(32, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(forward))  # 假设输出的时间步数与输入相同
    model.add(Reshape((forward, 1)))  # 重塑输出以匹配 [?, 10, 1]
    model.add(Activation("relu"))
    return model

def build_lstm_model2(input_shape, output_shape):
    model = Sequential()

    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_shape))
    model.add(Activation("relu"))

    return model

if __name__ == '__main__':
    # 准备数据
    DataName = "NGSIM_US101"
    backward = 50
    forward = 10

    train_x, val_x, _, train_y, val_y, _, _, _, _ = dt.load_data()
    train_x = train_x.reshape(train_x.shape[0], backward, 5)
    val_x = val_x.reshape(val_x.shape[0], backward, 5)

    # 加载模型
    model = build_lstm_model_light((backward, 5), forward)

    # 使用学习率衰减策略
    #lr_schedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9)
    lr_schedule = 0.001
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse')

    # 添加早停策略
    early_stopping = EarlyStopping(monitor='loss', patience=5, min_delta=0.00001, verbose=2)

    history = model.fit(train_x, train_y,
                        validation_data=(val_x, val_y),
                        epochs=1000, batch_size=64, verbose=2,
                        callbacks=[early_stopping])

    combined_loss_history = np.column_stack((history.history['loss'], history.history['val_loss']))
    os.makedirs(f'./results_{DataName}', exist_ok=True)
    np.savetxt(f"./results_{DataName}/convergence_rate.csv", combined_loss_history, delimiter=",", header="train_loss,val_loss")

    model.save(f"./model/{DataName}.h5")

    #plot_model(model, to_file='./results/model_plot IDM LSTM.png', show_shapes=True, show_layer_names=True)
