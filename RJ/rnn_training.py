import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
df = pd.read_csv("UNSW_NB15_training-set.csv")

# Step 2: Data preprocessing
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

cat_cols = [c for c in df.select_dtypes(include=['object']).columns]
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

df.fillna(0, inplace=True)

if 'label' in df.columns:
    X = df.drop(columns=['label'])
    y = df['label']
elif 'attack_cat' in df.columns:
    X = df.drop(columns=['attack_cat'])
    y = df['attack_cat'].apply(lambda x: 0 if x == le.fit_transform(['Normal'])[0] else 1)
else:
    raise ValueError("Dataset must contain 'label' or 'attack_cat' column.")

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Prepare data for RNN (3D input)
X_train_rnn = np.expand_dims(X_train.values, axis=2)
X_test_rnn = np.expand_dims(X_test.values, axis=2)

# Step 4: Build RNN model
model_rnn = Sequential([
    SimpleRNN(64, input_shape=(X_train_rnn.shape[1], 1), return_sequences=True),
    Dropout(0.2),
    SimpleRNN(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_rnn.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

# Step 5: Train RNN model
history_rnn = model_rnn.fit(X_train_rnn, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=1)

# Step 6: Predict and evaluate
y_pred_prob = model_rnn.predict(X_test_rnn).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
print(f"\n🔹 RNN Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.title("RNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"RNN ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – RNN Model")
plt.legend(loc="lower right")
plt.show()

plt.figure()
plt.plot(history_rnn.history['accuracy'], label='Training Accuracy')
plt.plot(history_rnn.history['val_accuracy'], label='Validation Accuracy')
plt.title("RNN Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
