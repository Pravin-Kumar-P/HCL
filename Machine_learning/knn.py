import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score

class KNNClassifier:
    def __init__(self, k):
        self.k = k
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((np.array(x1, dtype=float) - np.array(x2, dtype=float)) ** 2))
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            distances = []
            for i, train_point in enumerate(self.X_train):
                dist = self.euclidean_distance(test_point, train_point)
                distances.append((dist, self.y_train[i]))
            distances.sort(key=lambda x: x[0])
            k_nearest_neighbors = distances[:self.k]
            labels = [label for _, label in k_nearest_neighbors]
            predicted = Counter(labels).most_common(1)[0][0]
            predictions.append(predicted)
        return predictions
def load_data(file_path):
    df = pd.read_csv(file_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df
def main():
    file_path = input("Enter the CSV filename (with path if needed): ").strip()
    df = load_data(file_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Available numeric columns: {numeric_cols}") 
    selected_features = []
    num_features = int(input("How many numeric features do you want to use? "))
    for i in range(num_features):
        while True:
            col = input(f"Enter numeric column name {i + 1}: ").strip()
            if col in numeric_cols:
                selected_features.append(col)
                break
            else:
                print(f"Invalid column. Please choose from: {numeric_cols}")
    target_col = input("Enter name of categorical target column: ").strip()
    X = df[selected_features].astype(float).values
    y = df[target_col].values
    split_idx = int(0.7 * len(df))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    k = max(1, round(0.1 * len(X_train))) #10 percent k
    k = k + 1 if k % 2 == 0 else k 
    knn_classifier = KNNClassifier(k)
    knn_classifier.fit(X_train, y_train)

    preds = knn_classifier.predict(X_test)

    print("\nPerformance Metrics (Per Class):")
    classes = np.unique(y)
    print(f"{'Class':<12}{'Precision':<12}{'Recall':<12}{'F1 Score':<12}")
    for cls in classes:
        cls_precision = precision_score(y_test, preds, labels=[cls], average='macro', zero_division=0)
        cls_recall = recall_score(y_test, preds, labels=[cls], average='macro', zero_division=0)
        cls_f1 = f1_score(y_test, preds, labels=[cls], average='macro', zero_division=0)
        print(f"{cls:<12}{cls_precision:<12.2f}{cls_recall:<12.2f}{cls_f1:<12.2f}")

if __name__ == "__main__":
    main()
