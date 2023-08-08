import pandas as pd
import random
from IDM import IDM
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def monte_carlo_optimization(df, num_iterations):
    best_rmse = 100000
    best_arg = None

    # 使用tqdm显示进度条
    with tqdm(total=num_iterations, desc='Iterations', postfix={'Best RMSE': float('inf')}) as pbar:
        for _ in range(num_iterations):
            vf = random.uniform(22, 26)
            A = random.uniform(0, 1)
            b = random.uniform(2, 3)
            s0 = random.uniform(2, 5)
            T = random.uniform(0.5, 1.5)
            arg = (round(vf, 3), round(A, 3), round(b, 3), round(s0, 3), round(T, 3))

            df['a_hat'] = df.apply(lambda row: IDM(arg, row['v'], row['v'] - row['v-1'], row['y-1'] - row['y']),axis=1)
            #df['a_hat'] = df.apply(lambda row: IDM(arg, row['v'], row['v'] - row['v-1'], row['Space_Headway']), axis=1)
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
df = df[df['Space_Headway'] > 4] # 这个阈值直接决定了标定结果
df = df.dropna(subset=['v', 'v-1', 'Space_Headway'])
print('After filtering  len(df)=', len(df))

# 标定
best_arg, best_rmse = monte_carlo_optimization(df, num_iterations = 50)

# 结果保存

