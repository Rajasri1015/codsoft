import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Step 1: Load dataset
df = pd.read_csv("UNSW_NB15_training-set.csv")

# Step 2: Data preprocessing

# Remove 'id' column if present
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

# Encode categorical columns
cat_cols = [c for c in df.select_dtypes(include=['object']).columns]
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Fill missing values with 0
df.fillna(0, inplace=True)

# Features and labels split
if 'label' in df.columns:
    X = df.drop(columns=['label'])
    y = df['label']
elif 'attack_cat' in df.columns:
    X = df.drop(columns=['attack_cat'])
    # Convert attack_cat to binary: Normal=0, others=1
    y = df['attack_cat'].apply(lambda x: 0 if x == le.fit_transform(['Normal'])[0] else 1)
else:
    raise ValueError("Dataset must contain 'label' or 'attack_cat' column.")

# Scale features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Prepare data for CNN input
X_train_cnn = np.expand_dims(X_train.values, axis=2)
X_test_cnn = np.expand_dims(X_test.values, axis=2)

# Build CNN model
model_cnn = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile CNN
model_cnn.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

# Train CNN model
history_cnn = model_cnn.fit(X_train_cnn, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=1)

# Predictions and evaluation
y_pred_prob = model_cnn.predict(X_test_cnn).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

# Print accuracy and classification report
acc = accuracy_score(y_test, y_pred)
print(f"\n🔹 CNN Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"CNN ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – CNN Model")
plt.legend(loc="lower right")
plt.show()

# Plot training vs validation accuracy
plt.figure()
plt.plot(history_cnn.history['accuracy'], label='Training Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='Validation Accuracy')
plt.title("CNN Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

