import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Step 1: Load the Iris dataset
def load_iris():
    from sklearn.datasets import load_iris
    data = load_iris()
    return data.data, data.target

# Step 2: Split the dataset
def train_test_split(X, y, test_size=0.2):
    num_test = int(len(X) * test_size)
    indices = np.random.permutation(len(X))
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]

# Step 3: Implement K-NN
class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.linalg.norm(x_train - x) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Step 4: Evaluate the Model
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
    return cm

# Main workflow
def main():
    # Load data
    X, y = load_iris()

    # Split data
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    # Find the best K
    k_values = range(1, 21)
    accuracies = []

    for k in k_values:
        model = KNN(k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy(y_test, y_pred)
        accuracies.append(acc)

    # Best K
    best_k = k_values[np.argmax(accuracies)]
    print(f'Best K: {best_k} with Accuracy: {max(accuracies)}')

    # Plotting K vs Accuracy
    plt.figure()
    plt.plot(k_values, accuracies, marker='o')
    plt.title('K vs Accuracy')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid()
    plt.show()

    # Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()