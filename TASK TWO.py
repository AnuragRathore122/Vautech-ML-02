
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


df = pd.read_csv("C:\\Users\\Anurag Rathore\\Downloads\\bank.csv")
print("Initial shape:", df.shape)

# 3. Encode Categorical Columns

label_encoder = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = label_encoder.fit_transform(df[col].astype(str))


# 4. Handle Missing Values

imputer = SimpleImputer(strategy="median")
df[df.columns] = imputer.fit_transform(df)

# 5. Separate Features & Target

target_col = df.columns[-1]
X = df.drop(target_col, axis=1)
y = df[target_col]

print("\nClass distribution BEFORE outlier removal:")
print(y.value_counts())

# 6. Outlier Removal (ONLY on features)

Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1

mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)

X = X[mask]
y = y[mask]

print("\nClass distribution AFTER outlier removal:")
print(y.value_counts())

# 7. Handle Class Imbalance (Oversampling)

df_clean = pd.concat([X, y], axis=1)

majority = df_clean[df_clean[target_col] == df_clean[target_col].value_counts().idxmax()]
minority = df_clean[df_clean[target_col] == df_clean[target_col].value_counts().idxmin()]

minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

df_balanced = pd.concat([majority, minority_upsampled])

print("\nClass distribution AFTER balancing:")
print(df_balanced[target_col].value_counts())

# 8. Scaling

X_final = df_balanced.drop(target_col, axis=1)
y_final = df_balanced[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# 9. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_final, test_size=0.2, random_state=42, stratify=y_final
)

# 10. Model Training

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)

# 11. Evaluation

y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
