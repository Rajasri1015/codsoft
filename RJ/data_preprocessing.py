import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets - update the file paths as needed
train_df = pd.read_csv("UNSW_NB15_training-set.csv")
test_df = pd.read_csv("UNSW_NB15_testing-set.csv")

# Standardize column names by removing extra spaces
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# Columns to remove if they exist
cols_to_drop = ['srcip', 'dstip', 'attack_cat']

# Print columns before dropping
print("Training columns before drop:", train_df.columns.tolist())
print("Testing columns before drop:", test_df.columns.tolist())

# Drop columns safely without errors if they don't exist
train_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
test_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Identify categorical columns
cat_cols = train_df.select_dtypes(include=['object']).columns
print("Categorical columns identified:", list(cat_cols))

# Encode categorical columns safely
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    # Map test data, assign -1 for unseen categories
    test_df[col] = test_df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    encoders[col] = le

# Separate features and labels
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Normalize features with MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verify data shapes and target distribution
print("\nPost-processing information:")
print("X_train shape:", X_train_scaled.shape)
print("X_test shape:", X_test_scaled.shape)
print("Training labels distribution:\n", y_train.value_counts())

# Plot feature correlation heatmap (top 15 features)
corr_matrix = pd.DataFrame(X_train, columns=X_train.columns).corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix.iloc[:15, :15], cmap="coolwarm")
plt.title("Correlation Heatmap of Top 15 Features")
plt.show()

# Plot label distribution
plt.figure(figsize=(5,4))
sns.countplot(x=y_train, palette=['skyblue', 'salmon'])
plt.title("Label Distribution After Preprocessing")
plt.xlabel("Label (0 = Normal, 1 = Attack)")
plt.ylabel("Record Count")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
