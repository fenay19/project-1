import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    df.dropna(inplace=True)

    le = LabelEncoder()
    for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
        df[col] = le.fit_transform(df[col])

    df = pd.get_dummies(df, columns=['Dependents'], drop_first=True)

    X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = df['Loan_Status']
    return X, y
