import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Example accuracy results from previous steps
algorithms = ['CNN', 'LSTM', 'RNN', 'SVM', 'Logistic Regression', 'Naive Bayes']
accuracies = [94.7, 96.4, 95.1, 91.6, 88.9, 89.5]

# Example Federated Accuracy over rounds
fed_rounds = [1, 2, 3, 4, 5]
fed_acc = [92.4, 93.7, 94.6, 95.5, 96.1]

# 1️⃣ Federated Learning Accuracy Graph
plt.figure(figsize=(7, 5))
plt.plot(fed_rounds, fed_acc, marker='o', color='teal', linewidth=2)
plt.title("Federated Learning: Global Accuracy per Round")
plt.xlabel("Federated Rounds")
plt.ylabel("Global Model Accuracy (%)")
plt.grid(True)
plt.show()

# 2️⃣ Algorithm Accuracy Comparison (Bar Chart)
plt.figure(figsize=(8, 5))
bars = plt.bar(algorithms, accuracies, color=['#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#E53935', '#3949AB'])
plt.title("Algorithm Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(80, 100)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.5, f"{acc:.1f}%", ha='center', fontsize=9)
plt.show()

# 3️⃣ ROC Curves for CNN, LSTM, RNN (example data)
fpr_cnn = [0, 0.05, 0.1, 0.2, 1]
tpr_cnn = [0, 0.91, 0.94, 0.96, 1]
fpr_lstm = [0, 0.03, 0.08, 0.15, 1]
tpr_lstm = [0, 0.93, 0.96, 0.98, 1]
fpr_rnn = [0, 0.04, 0.09, 0.17, 1]
tpr_rnn = [0, 0.91, 0.95, 0.97, 1]

plt.figure(figsize=(7, 5))
plt.plot(fpr_cnn, tpr_cnn, label='CNN (AUC=0.97)')
plt.plot(fpr_lstm, tpr_lstm, label='LSTM (AUC=0.98)')
plt.plot(fpr_rnn, tpr_rnn, label='RNN (AUC=0.97)')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – Deep Models")
plt.legend(loc="lower right")
plt.show()

# 4️⃣ Final Confusion Matrix for Best Model (LSTM)
cm = np.array([[11120, 130], [280, 11040]])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("LSTM Confusion Matrix (Final Global Model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
