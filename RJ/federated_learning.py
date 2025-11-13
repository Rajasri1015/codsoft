import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step 1: Load and preprocess dataset
df = pd.read_csv("UNSW_NB15_training-set.csv")
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)
cat_cols = [col for col in df.select_dtypes(include=['object']).columns]
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))
df.fillna(0, inplace=True)

if 'label' in df.columns:
    X = df.drop(columns=['label'])
    y = df['label']
elif 'attack_cat' in df.columns:
    X = df.drop(columns=['attack_cat'])
    y = df['attack_cat'].apply(lambda x: 0 if x == le.transform(['Normal'])[0] else 1)
else:
    raise ValueError("Dataset must have 'label' or 'attack_cat' column.")

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Step 2: Shard data into clients
def shard_data(X, y, num_clients=5):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    X_shards = [X.iloc[idx] for idx in splits]
    y_shards = [y.iloc[idx] for idx in splits]
    return X_shards, y_shards

X_shards, y_shards = shard_data(X_train, y_train, num_clients=5)

# Step 3: Build LSTM model function
def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Prepare data shape for LSTM
X_test_lstm = np.expand_dims(X_test.values, axis=2)
input_shape = (X_test_lstm.shape[1], 1)

# Step 4: Federated training simulation with FedAvg
def federated_train_and_aggregate(model_fn, X_shards, y_shards, rounds=5, local_epochs=1):
    global_model = model_fn(input_shape)
    global_weights = global_model.get_weights()
    global_acc = []

    for r in range(rounds):
        print(f"\n🔸 Round {r+1} started...")
        local_weights = []
        for i in range(len(X_shards)):
            local_model = clone_model(global_model)
            local_model.set_weights(global_weights)
            X_local = np.expand_dims(X_shards[i].values, axis=2)
            y_local = y_shards[i].values
            local_model.fit(X_local, y_local, epochs=local_epochs, batch_size=64, verbose=0)
            local_weights.append(local_model.get_weights())
            print(f"Client {i+1} trained locally.")

        # Average weights (FedAvg)
        new_weights = [np.mean(np.array([w[i] for w in local_weights]), axis=0) for i in range(len(global_weights))]
        global_model.set_weights(new_weights)
        global_weights = new_weights

        # Evaluate global model on test data
        y_pred_prob = global_model.predict(X_test_lstm).ravel()
        y_pred = (y_pred_prob > 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        print(f"🔹 Global Accuracy after Round {r+1}: {acc*100:.2f}%")
        global_acc.append(acc*100)

    return global_acc

# Run federated learning simulation
fed_acc = federated_train_and_aggregate(build_lstm, X_shards, y_shards, rounds=5, local_epochs=1)

# Step 5: Plot accuracy per round
plt.figure(figsize=(7,5))
plt.plot(range(1, len(fed_acc)+1), fed_acc, marker='o', color='teal')
plt.title("Federated Learning – Global Accuracy per Round")
plt.xlabel("Round")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.show()
