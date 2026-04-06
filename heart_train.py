import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. LOAD YOUR EXTRACTED FILE
print("--- Loading Heart Disease Data ---")
df = pd.read_csv('heart_disease_uci.csv.csv')

# 2. CLEANING DATA
# Drop ID and Dataset as they don't help prediction
df = df.drop(['id', 'dataset'], axis=1)

# Handle Missing Values (Important for this specific dataset)
# Fill numeric columns with Median
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical columns with the most frequent value (Mode)
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 3. ENCODING (Changing Text to Numbers)
# We convert Sex, CP, etc. into numbers so the AI can read them
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le # Save encoders for the web app later

# Simplify the Target: 0 = Healthy, 1+ = Heart Disease
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# 4. PREPARE FEATURES (X) AND TARGET (y)
X = df.drop('num', axis=1)
y = df['num']

# 5. SPLIT AND SCALE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. TRAIN THE AI
print("--- Training PulseGuard AI (Logistic Regression) ---")
model = LogisticRegression()
model.fit(X_train, y_train)

# 7. CHECK PERFORMANCE
y_pred = model.predict(X_test)
print(f"\nDiagnostic Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# 8. SAVE THE BRAIN, SCALER, AND ENCODERS
joblib.dump(model, 'heart_model.pkl')
joblib.dump(scaler, 'heart_scaler.pkl')
joblib.dump(le_dict, 'heart_encoders.pkl') # We save all text-to-number rules

print("\n✅ SUCCESS: 'heart_model.pkl' and 'heart_scaler.pkl' created!")