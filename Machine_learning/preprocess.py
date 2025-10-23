import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv("train.csv")

# Display sample
print("--- Initial Data Sample ---")
print(df.head())

# Check missing values
print("\n--- Missing Values Count ---")
print(df.isna().sum())

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Cabin'] = df['Cabin'].fillna('Unknown')   # handle Cabin

print("\n--- After Filling Missing Values ---")
print(df.isna().sum())

# Select features and target
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = df['Survived']

# Encode categorical columns safely
le = LabelEncoder()
X.loc[:, 'Sex'] = le.fit_transform(X['Sex'])
X.loc[:, 'Embarked'] = le.fit_transform(X['Embarked'])

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Preprocessing Complete ---")
print("Training Data Shape:", X_train_scaled.shape)
print("Testing Data Shape:", X_test_scaled.shape)
