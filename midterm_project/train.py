import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle

MODEL_NAME = 'Logistic Regression'
BEST_PARAMS = {
    'C': 0.01, 
    'solver': 'liblinear',
    'random_state': 42
}


output_file = 'heart_disease_model.pkl'

df = pd.read_csv('heart_disease.csv')
df.drop('Alcohol Consumption', axis=1, inplace=True)

# replace numerical null with median
numerical_cols = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level', 'Sleep Hours']
for col in numerical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# replace categorical null with mode
categorical_cols = ['Gender', 'Exercise Habits', 'Smoking', 'Family Heart Disease', 'Diabetes', 'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol', 'Stress Level', 'Sugar Consumption']
for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])


df.rename(columns={'Heart Disease Status': 'status'}, inplace=True)
df['status'] = (df['status'] == 'Yes').astype(int)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_full_train = df_full_train.reset_index(drop=True)

y_full_train = df_full_train.status.values
del df_full_train['status']

print(f"Training the final model: {MODEL_NAME}")
dv = DictVectorizer(sparse=False)
train_dicts = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(train_dicts)

model = LogisticRegression(
    C = 0.01, 
    solver = 'liblinear',
    random_state = 42
)

model.fit(X_full_train, y_full_train)

model_artifacts = {
    'vectorizer': dv,
    'model': model
}

with open(output_file, 'wb') as f_out:
    pickle.dump(model_artifacts, f_out)

print(f"Model artifacts successfully saved to '{output_file}'")