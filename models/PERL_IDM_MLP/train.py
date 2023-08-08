import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import Callback
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam
import data as dt


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
            if len(self.loss_history) > self.patience and abs(
                    loss - self.loss_history[-self.patience - 1]) < self.threshold:
                print("Converged.")
                self.model.stop_training = True


if __name__ == '__main__':
    # 准备数据
    train_x, _, train_y, _, _, _, _ = dt.load_data()

    # 构建 MLP 模型
    model = Sequential([
        Dense(128, activation='relu', input_shape=(train_x.shape[1],)),
        Dense(128, activation='relu'),
        Dense(train_y.shape[1])
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    print(model.input_shape)
    print(model.output_shape)

    convergence_checker = ConvergenceChecker(patience=5, threshold=0.0001)

    history = model.fit(train_x, train_y, epochs=50, batch_size=64, verbose=2, callbacks=[convergence_checker])
    loss_history = convergence_checker.loss_history

    model.save("./model/NGSIM.h5")
    np.savetxt("./results/convergence_rate.csv", loss_history, delimiter=",")

    #plot_model(model, to_file='./results/model_structure PERL IDM_MLP.png', show_shapes=True, show_layer_names=True)
