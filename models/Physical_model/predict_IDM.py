'''
IDM有两种计算模式，
第一种是评价IDM的预测能力，在每个sample内，对forward部分进行累计预测，即使用上一个时刻的IDM预测值作为输入，以保证与其他模型一致
第二种是计算residual，作为PERL模型的输入，是对所有数据进行单步预测，即使用上一个时刻的真实值作为输入
'''

import pandas as pd
import numpy as np
from IDM import IDM
from tqdm import tqdm

arg = (22.39, 0.06, 2.9, 3.91, 0.6)
backward = 50
forward = 10
t_chain = backward + forward


# 读取数据，形成chain
import sys
sys.path.append('/home/ubuntu/Documents/PERL/models')  # 将 load_data.py 所在的目录添加到搜索路径
import load_data_fun
df = load_data_fun.load_data()


filtered_data = []
chain_id = 0  # 初始化子序列编号
for _, group in tqdm(df.groupby(['id', 'Lane_ID']), total=df['id'].nunique()):
    group = group.sort_values(by='t')  # 按时间排序
    for i in range(0,len(group) - t_chain,t_chain):
        sub_group = group.iloc[i:i+t_chain].copy()  # 选取子序列并创建副本
        sub_group.reset_index(drop=True, inplace=True)  # 重置索引
        sub_group['chain_id'] = chain_id  # 添加子序列编号

        # 第一种计算模式，计算 a，v，y
        for j in range(backward, t_chain):
            if j == backward:
                sub_group.loc[j, 'a_IDM'] = IDM(arg,
                                                sub_group.loc[j-1, 'v'],
                                                sub_group.loc[j-1, 'v']-sub_group.loc[j-1, 'v-1'],
                                                sub_group.loc[j-1, 'y-1']-sub_group.loc[j-1, 'y'])
                sub_group.loc[j, 'v_IDM'] = sub_group.loc[j-1, 'v'] + 0.1 * sub_group.loc[j-1, 'a']
                sub_group.loc[j, 'y_IDM'] = sub_group.loc[j-1, 'y'] + 0.1 * sub_group.loc[j-1, 'v'] + 0.5 * 0.01 * sub_group.loc[j-1, 'a']
            else:
                sub_group.loc[j, 'a_IDM'] = IDM(arg,
                                                sub_group.loc[j-1, 'v_IDM'],
                                                sub_group.loc[j-1, 'v_IDM']-sub_group.loc[j-1, 'v-1'],
                                                sub_group.loc[j-1, 'y-1']-sub_group.loc[j-1, 'y_IDM'])
                sub_group.loc[j, 'v_IDM'] = sub_group.loc[j-1, 'v_IDM'] + 0.1 * sub_group.loc[j-1, 'a_IDM']
                sub_group.loc[j, 'y_IDM'] = sub_group.loc[j-1, 'y_IDM'] + 0.1 * sub_group.loc[j-1, 'v_IDM'] + 0.5 * 0.01 * sub_group.loc[j-1, 'a_IDM']
        sub_group['a_IDM'] = sub_group['a_IDM'].astype(np.float32).round(3)
        sub_group['v_IDM'] = sub_group['v_IDM'].astype(np.float32).round(3)
        sub_group['y_IDM'] = sub_group['y_IDM'].astype(np.float32).round(3)

        # 第二计算模式，计算 a_residual
        sub_group.loc[0, 'a_residual_IDM'] = 0
        for j in range(1, t_chain):
            sub_group.loc[j, 'a_IDM_2'] = IDM(arg,
                                              sub_group.loc[j - 1, 'v'],
                                              sub_group.loc[j - 1, 'v'] - sub_group.loc[j - 1, 'v-1'],
                                              max(0, sub_group.loc[j - 1, 'y-1'] - sub_group.loc[j - 1, 'y']) )
            sub_group.loc[j, 'a_residual_IDM'] = round(sub_group.loc[j, 'a_IDM_2'] - sub_group.loc[j, 'a'], 3)

        filtered_data.append(sub_group)
        chain_id += 1
filtered_data_df = pd.concat(filtered_data)
filtered_data_df.to_csv("/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_IDM_results.csv", index=False)