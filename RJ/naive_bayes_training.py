import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset (update path if needed)
df = pd.read_csv("UNSW_NB15_training-set.csv")

# Step 2: Data preprocessing
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
    y = df['attack_cat'].apply(lambda x: 0 if x == le.fit_transform(['Normal'])[0] else 1)
else:
    raise ValueError("Dataset must contain 'label' or 'attack_cat' column.")

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Train Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Step 4: Predictions and evaluation
y_pred_nb = nb_model.predict(X_test)
y_prob_nb = nb_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred_nb)
print(f"\n🔹 Naive Bayes Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred_nb))

# Confusion Matrix visualization
cm = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve visualization
fpr, tpr, _ = roc_curve(y_test, y_prob_nb)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"Naive Bayes ROC (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Naive Bayes Model")
plt.legend(loc="lower right")
plt.show()
