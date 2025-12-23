import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    df.drop(columns=['Customer_ID'], inplace=True)

    categorical_cols = df.select_dtypes(include='object').columns
    numeric_features = [
        'Applicant_Income',
        'Coapplicant_Income',
        'Loan_Amount',
        'Loan_Amount_Term'
    ]

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    df.to_csv(output_path, index=False)
    print("Preprocessing selesai:", output_path)

if __name__ == "__main__":
    preprocess_data(
        input_path="Loan_Eligibility_raw.csv",
        output_path="preprocessing/Loan_Eligibility_preprocessing.csv"
    )