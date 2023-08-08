import numpy as np
from keras.models import load_model
import data as dt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import argparse


if __name__ == '__main__':

    train_x, test_x, train_y, test_y_real, rows_test, A_min, A_max = dt.load_data()
    model = load_model("./model/NGSim.h5")
    test_y_predict = model.predict(test_x)

    A_real = test_y_real.tolist()
    A_hat  = test_y_predict.tolist()
    A_real = A_real * (A_max - A_min) + A_min
    A_hat  = A_hat * (A_max - A_min) + A_min

    mse = mean_squared_error(A_real, A_hat)
    print('MSE when predicting acceleration:', mse)

    x = np.arange(test_y_predict.shape[0])
    plt.figure(figsize=(10, 4))
    plt.plot(x, A_real, '.', color='b', markersize=1, label='Original a')
    plt.plot(x, A_hat, '.', color='r', markersize=1, label='LSTM Predicted a')
    plt.xlabel('Index')
    plt.ylabel('Acceleration error (m/s^2)')
    plt.ylim(-4, 4)
    #plt.title('MSE: {:.4f}'.format(mse))
    plt.legend()
    plt.savefig('../Results/NGSim/LSTM_result.png')
    plt.show()


