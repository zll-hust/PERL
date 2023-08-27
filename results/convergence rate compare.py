import numpy as np
import matplotlib.pyplot as plt

def pad_data(data, max_length):
    """用数据的最后一个值填充数据，直到其长度与max_length一致"""
    return np.pad(data, (0, max_length - len(data)), 'edge')

DataName = "NGSIM_US101"

# 加载MLP和LSTM的convergence rate数据
# d="_fulldata"
# d="_data500"
# d="_data5000"
d=""
Data_driven_LSTM_data = np.loadtxt(f"../models/Data_driven_LSTM/results_{DataName}/convergence_rate{d}.csv", delimiter=",")
PINN_IDM_LSTM_data = np.loadtxt(f"../models/PINN_IDM_LSTM/results_{DataName}/convergence_rate{d}.csv", delimiter=",")
PERL_IDM_LSTM_data = np.loadtxt(f"../models/PERL_IDM_LSTM/results_{DataName}/convergence_rate{d}.csv", delimiter=",")

max_length = max(len(Data_driven_LSTM_data), len(PINN_IDM_LSTM_data), len(PERL_IDM_LSTM_data))

# 使用pad_data函数确保所有数据的长度与max_length一致
Data_driven_LSTM_data = pad_data(Data_driven_LSTM_data, max_length)
PINN_IDM_LSTM_data = pad_data(PINN_IDM_LSTM_data, max_length)
PERL_IDM_LSTM_data = pad_data(PERL_IDM_LSTM_data, max_length)

fig, ax = plt.subplots(figsize=(6, 4))

# 橘色系
ax.plot(Data_driven_LSTM_data[:, 0], label="Data-driven (LSTM) Train Loss", color="#FFA500")
ax.plot(Data_driven_LSTM_data[:, 1], label="Data-driven (LSTM) Val Loss", linestyle='--', color="#ff7700")

# 紫色系
ax.plot(PINN_IDM_LSTM_data[:, 0], label="PINN (IDM_LSTM) Train Loss", color="#9933FF")
ax.plot(PINN_IDM_LSTM_data[:, 1], label="PINN (IDM_LSTM) Val Loss", linestyle='--', color="#7A00CC")

# 蓝色系
ax.plot(PERL_IDM_LSTM_data[:, 0], label="PERL (IDM_LSTM) Train Loss", color="#0073e6")
ax.plot(PERL_IDM_LSTM_data[:, 1], label="PERL (IDM_LSTM) Val Loss", linestyle='--', color="#0059b3")

# 其他设置
ax.set_ylabel("MSE Loss $(m^2/s^4)$")
ax.set_yscale("log")  # 设置y轴为对数尺度
plt.ylim(5e-4, 10)

plt.xlabel("Epoch")
plt.xlim(0, max_length)

plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)

plt.legend(loc='upper right', frameon=False)

plt.savefig(f'Convergence Rate_data5000_lstm32.png')
plt.show()
