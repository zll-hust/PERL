import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


backward = 50
forward = 10
t_chain = backward + forward


# Data preprocessing
def preprocess_data(look_back=50, look_forward=30):
    df = pd.read_csv("/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_IDM_results.csv")
    print(df.shape)

    # Initialize the scaler
    scaler_y_1 = MinMaxScaler()
    scaler_y   = MinMaxScaler()
    scaler_v_1 = MinMaxScaler()
    scaler_v   = MinMaxScaler()
    scaler_a   = MinMaxScaler()
    scaler_a_residual_IDM = MinMaxScaler()

    # Fit the scaler on the entire dataset
    scaler_y_1.fit(df['y-1'].values.reshape(-1, 1))
    scaler_y.fit(df['y'].values.reshape(-1, 1))
    scaler_v_1.fit(df['v-1'].values.reshape(-1, 1))
    scaler_v.fit(df['v'].values.reshape(-1, 1))
    scaler_a.fit(df['a'].values.reshape(-1, 1))
    scaler_a_residual_IDM.fit(df['a'].values.reshape(-1, 1))

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
        A_residual_IDM_normalized = scaler_a_residual_IDM.transform(chain_df['a_residual_IDM'].values.reshape(-1, 1))

        # Create the feature vectors and targets for each sample in this chain
        for i in range(0, len(chain_df) - backward - forward + 1, backward + forward):
            X_sample = np.concatenate((Y_1_normalized[i+1:i + backward, 0],
                                       Y_normalized[i+1:i + backward, 0],
                                       V_1_normalized[i+1:i + backward, 0],
                                       V_normalized[i+1:i + backward, 0],
                                       A_normalized[i+1:i + backward, 0]), axis=0)
                                       #A_residual_IDM_normalized[i+1:i + backward, 0] ), axis=0)
            Y_sample = A_residual_IDM_normalized[i + backward:i + backward + forward, 0]

            X.append(X_sample)
            Y.append(Y_sample)

    # Convert the lists to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, a_min, a_max

# Load the data
data_path = "/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_IDM_results.csv"
df = pd.read_csv(data_path)
X_train, X_test, y_train, y_test, a_min, a_max= preprocess_data(df)

# Model architecture
lstm_units = 50
dropout_rate = 0.2
l2_reg = 0.01
input_shape = (X_train.shape[1],)
#input_shape = (backward, len(features))

model = Sequential([
    LSTM(lstm_units, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(l2_reg)),
    Dropout(dropout_rate),
    LSTM(lstm_units, kernel_regularizer=l2(l2_reg)),
    Dropout(dropout_rate),
    Dense(y_train.shape[1])
])
optimizer = Adam(learning_rate=0.05)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Training strategy
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=64, callbacks=[early_stopping, reduce_lr])
