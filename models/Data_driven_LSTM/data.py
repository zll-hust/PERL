import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import glob


backward = 50
forward = 10
t_chain = backward + forward

def load_data(look_back=50, look_forward=30):
    df = pd.read_csv("/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_IDM_results.csv")

    # Initialize the scaler
    scaler_y_1 = MinMaxScaler()
    scaler_y   = MinMaxScaler()
    scaler_v_1 = MinMaxScaler()
    scaler_v   = MinMaxScaler()
    scaler_a   = MinMaxScaler()

    # Fit the scaler on the entire dataset
    scaler_y_1.fit(df['y-1'].values.reshape(-1, 1))
    scaler_y.fit(df['y'].values.reshape(-1, 1))
    scaler_v_1.fit(df['v-1'].values.reshape(-1, 1))
    scaler_v.fit(df['v'].values.reshape(-1, 1))
    scaler_a.fit(df['a'].values.reshape(-1, 1))

    a_min = min(df['a'])
    a_max = max(df['a'])
    print(f'a_min = {a_min}, a_max = {a_max}')

    # Initialize the lists to hold the features and targets
    X = []
    Y = []

    # Identify the unique chain IDs
    chain_ids = df['chain_id'].unique()

    for chain_id in chain_ids:
        # Get the subset of the DataFrame for this chain ID
        chain_df = df[df['chain_id'] == chain_id]

        # Normalize the features
        Y_1_normalized = scaler_y_1.transform(chain_df['y-1'].values.reshape(-1, 1))
        Y_normalized   = scaler_y.transform(chain_df['y'].values.reshape(-1, 1))
        V_1_normalized = scaler_v_1.transform(chain_df['v-1'].values.reshape(-1, 1))
        V_normalized   = scaler_v.transform(chain_df['v'].values.reshape(-1, 1))
        A_normalized   = scaler_a.transform(chain_df['a'].values.reshape(-1, 1))

        # Create the feature vectors and targets for each sample in this chain
        for i in range(0, len(chain_df) - backward - forward + 1, backward + forward):
            X_sample = np.concatenate((Y_1_normalized[i:i + backward, 0],
                                       Y_normalized[i:i + backward, 0],
                                       V_1_normalized[i:i + backward, 0],
                                       V_normalized[i:i + backward, 0],
                                       A_normalized[i:i + backward, 0]), axis=0)
            Y_sample = A_normalized[i + backward:i + backward + forward, 0]

            X.append(X_sample)
            Y.append(Y_sample)

    # Convert the lists to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, a_min, a_max