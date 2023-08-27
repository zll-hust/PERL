import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import glob

backward = 50
forward = 10
t_chain = backward + forward

def load_data(backward=50, forward=10):
    # df = pd.read_csv("/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_IDM_results.csv")
    df = pd.read_csv("/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_IDM_results.csv")

    df['delta_y'] = df['y-1']-df['y']

    # Initialize the scaler
    # scaler_y_1 = MinMaxScaler()
    # scaler_y   = MinMaxScaler()
    scaler_delta_y = MinMaxScaler()
    scaler_v_1 = MinMaxScaler()
    scaler_v   = MinMaxScaler()
    scaler_a   = MinMaxScaler()
    scaler_a_residual_IDM = MinMaxScaler()

    # Fit the scaler on the entire dataset
    # scaler_y_1.fit(df['y-1'].values.reshape(-1, 1))
    # scaler_y.fit(df['y'].values.reshape(-1, 1))
    scaler_delta_y.fit(df['delta_y'].values.reshape(-1, 1))
    scaler_v_1.fit(df['v-1'].values.reshape(-1, 1))
    scaler_v.fit(df['v'].values.reshape(-1, 1))
    scaler_a.fit(df['a'].values.reshape(-1, 1))
    scaler_a_residual_IDM.fit(df['a_residual_IDM'].values.reshape(-1, 1))

    a_residual_IDM_min = min(df['a_residual_IDM'])
    a_residual_IDM_max = max(df['a_residual_IDM'])
    print(f'a_residual_IDM_min = {a_residual_IDM_min}, a_residual_IDM_max = {a_residual_IDM_max}')

    # Initialize the lists to hold the features and targets
    X = []
    Y = []

    # Initialize the list to hold the chain_ids for each sample
    sample_chain_ids = []

    chain_ids = df['chain_id'].unique()
    for chain_id in chain_ids:
        # Get the subset of the DataFrame for this chain ID
        chain_df = df[df['chain_id'] == chain_id]

        # Normalize the features
        # Y_1_normalized = scaler_y_1.transform(chain_df['y-1'].values.reshape(-1, 1))
        # Y_normalized   = scaler_y.transform(chain_df['y'].values.reshape(-1, 1))
        delta_Y_normalized = scaler_delta_y.transform(chain_df['y'].values.reshape(-1, 1))
        V_1_normalized = scaler_v_1.transform(chain_df['v-1'].values.reshape(-1, 1))
        V_normalized   = scaler_v.transform(chain_df['v'].values.reshape(-1, 1))
        A_normalized   = scaler_a.transform(chain_df['a'].values.reshape(-1, 1))
        A_residual_IDM_normalized = scaler_a_residual_IDM.transform(chain_df['a_residual_IDM'].values.reshape(-1, 1))

        # Create the feature vectors and targets for each sample in this chain
        for i in range(0, len(chain_df) - backward - forward + 1, backward + forward):
            X_sample = np.concatenate((delta_Y_normalized[i:i + backward, 0],
                                       V_1_normalized[i:i + backward, 0],
                                       V_normalized[i:i + backward, 0],
                                       A_normalized[i:i + backward, 0],
                                       A_residual_IDM_normalized[i:i + backward, 0] ), axis=0)
            Y_sample = A_residual_IDM_normalized[i + backward:i + backward + forward, 0]

            X.append(X_sample)
            Y.append(Y_sample)
            sample_chain_ids.append(chain_id)

    # Convert the lists to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    print(f"Original number of samples: {len(X)}")


    # 随机选取一定数量的数据
    num_samples = 1000000
    np.random.seed(40)
    # If the total number of samples is greater than desired number, then choose a subset
    if len(X) > num_samples:
        indices = np.random.choice(len(X), num_samples, replace=False)
        X = X[indices]
        Y = Y[indices]
        sample_chain_ids = [sample_chain_ids[i] for i in indices]


    # divided into a training + validation set and a test set
    X_temp, X_test, y_temp, y_test, temp_chain_ids, test_chain_ids = train_test_split(X, Y, sample_chain_ids,
                                                                                      test_size=0.2, random_state=42)
    # divided into training set and validation set
    X_train, X_val, y_train, y_val, train_chain_ids, val_chain_ids = train_test_split(X_temp, y_temp, temp_chain_ids,
                                                                                      test_size=0.25, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test, a_residual_IDM_min, a_residual_IDM_max, test_chain_ids
