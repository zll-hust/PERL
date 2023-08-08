from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Embedding, LSTM, Dropout, Activation
from keras.callbacks import Callback
import data as dt
import argparse
import matplotlib.pyplot as plt
import numpy as np
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


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


def train_model(train_x, train_y, epochs, batch_size, dropout=0.2):
    model = Sequential()
    model.add(LSTM(256,
                   input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=True))
    model.add(Dropout(dropout))

    model.add(LSTM(256,
                   return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(train_y.shape[1]))  # 10个输出的全连接层
    model.add(Activation("relu"))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    convergence_checker = ConvergenceChecker(patience=3, threshold=0.0001)

    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2)
    loss_history = history.history['loss']
    np.savetxt("convergence_rate.csv", loss_history, delimiter=",")

    # 可视化损失函数值的变化
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('./results/training_loss LSTM.png')
    plt.show()

    return model


with tf.device('/gpu:0'):
    # 准备数据
    train_x, _, train_y, _, _, _ = dt.load_data()
    print("Shape of train_x:", train_x.shape[0], train_x.shape[1])
    print("Shape of train_y:", train_y.shape)
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
    train_y = train_y.reshape(train_y.shape[0], train_y.shape[1], 1)

    model = train_model(train_x, train_y, epochs=20, batch_size=64)
    model_name = "./model/NGSIM.h5"
    model.save(model_name)
    print('model trained and saved')

    # 保存模型结构图
    # plot_model(model, to_file='model_structure LSTM.png', show_shapes=True, show_layer_names=True)


