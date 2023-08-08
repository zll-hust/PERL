import pandas as pd
import random
from FVD import FVD
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def monte_carlo_optimization(df, num_iterations):
    best_rmse = 100000
    best_arg = None

    # 使用tqdm显示进度条
    with tqdm(total=num_iterations, desc='Iterations', postfix={'Best MSE': float('inf')}) as pbar:
        for _ in range(num_iterations):
            alpha = random.uniform(0.1, 2)
            lamda = random.uniform(0, 3)
            v_0 = random.uniform(15, 30)
            b = random.uniform(5, 25)
            beta = random.uniform(2, 8)
            arg = (round(alpha, 3), round(lamda, 3), round(v_0, 3), round(b, 3), round(beta, 3))

            df['a_hat'] = df.apply(lambda row: FVD(arg, row['v'], row['v'] - row['v-1'], row['y-1']-row['y']),axis=1)
            df['a_error'] = df['a_hat'] - df['a']

            mse = mean_squared_error(df['a'], df['a_hat'])
            rmse = np.sqrt(mse)
            if rmse < best_rmse:
                best_rmse = rmse
                best_arg = arg

            # 更新最小MSE的值
            pbar.set_postfix_str({'Best RMSE': round(best_rmse, 3), 'best_arg': best_arg})
            pbar.update(1)

    # plt.hist(df['A_error'], bins=20, color='blue', alpha=0.5)
    # plt.title('A_error Distribution')
    return best_arg, best_rmse


# Load data
import sys
sys.path.append('/home/ubuntu/Documents/PERL/models')  # 将 load_data.py 所在的目录添加到搜索路径
import load_data_fun
df = load_data_fun.load_data()

# 筛选
print('Before filtering len(df)=', len(df))
df = df[df['Preceding'] != 0]
df = df[df['Space_Headway'] > 5] # 这个阈值直接决定了标定结果
df = df.dropna(subset=['v', 'v-1', 'Space_Headway'])
print('After filtering  len(df)=', len(df))

# 标定
best_arg, best_rmse = monte_carlo_optimization(df, num_iterations = 50)

# 结果保存
# {'Best MSE': 24.108, 'best_arg': (0.3385, 2.6314, 19.8523, 16.4044, 5.0356)}]
# {'Best MSE': 46.61, 'best_arg': (0.8576, 1.6746, 26.7355, 14.4534, 4.1795)}]
# {'Best MSE': 813.892, 'best_arg': (4.1068, 2.7376, 15.4216, 8.7914, 5.1852)}]