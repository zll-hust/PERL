from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.callbacks import Callback
import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam

import sys
sys.path.append('/home/ubuntu/Documents/PERL/models/')
from PERL_IDM_MLP import data as dt


class ConvergenceChecker(Callback):
    def __init__(self, patience=5, threshold=0.0001):
        super(ConvergenceChecker, self).__init__()
        self.patience = patience
        self.threshold = threshold
        self.loss_history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss:
            self.loss_history.append(loss)
            if len(self.loss_history) > self.patience and abs(
                    loss - self.loss_history[-self.patience - 1]) < self.threshold:
                print("Converged.")
                self.model.stop_training = True


if __name__ == '__main__':
    # 准备数据
    backward = 50
    train_x, _, train_y, _, _, _, _ = dt.load_data()
    train_x = train_x.reshape(train_x.shape[0], backward, 6)

    # 构建LSTM模型
    model = Sequential()
    model.add(LSTM(256,
                   input_shape=(backward, 6),
                   return_sequences=False))
    model.add(Dropout(rate=0.05))
    model.add(Dense(train_y.shape[1]))  # 10个输出的全连接层
    model.add(Activation("relu"))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    convergence_checker = ConvergenceChecker(patience=5, threshold=0.0001)

    history = model.fit(train_x, train_y, epochs=50, batch_size=64, verbose=0,
                        callbacks=[convergence_checker])
    loss_history = convergence_checker.loss_history
    np.savetxt("./results/convergence_rate.csv", loss_history, delimiter=",")

    model.save("./model/NGSIM.h5")

    #plot_model(model, to_file='./results/model_plot IDM LSTM.png', show_shapes=True, show_layer_names=True)
