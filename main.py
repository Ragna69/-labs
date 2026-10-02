import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

try:
    df = pd.read_csv('titanic.csv')
except FileNotFoundError:
    print("Ошибка: Файл 'titanic.csv' не найден.")
    df = pd.DataFrame()

if not df.empty:
    print("Первые 5 строк датасета")
    print(df.head())

    print("\n Общая информация о датасете")
    print(df.info())

    print("\n 3: Количество пропущенных значений")
    missing_values = df.isnull().sum()
    print(missing_values)

    print("\n 4: Заполнение пропусков ")

    cols_with_missing = df.columns[df.isnull().any()].tolist()

    for col in cols_with_missing:
        if df[col].dtype == 'object':
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"Столбец '{col}' заполнен модой: {mode_val}")
        else:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Столбец '{col}' заполнен медианой: {median_val}")

    print("\n Проверка после заполнения (должно быть 0 везде):")
    print(df.isnull().sum())

    print("\n 5: Нормализация данных")

    numeric_cols = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']

    existing_numeric = [col for col in numeric_cols if col in df.columns]

    if existing_numeric:
        scaler = MinMaxScaler()
        df[existing_numeric] = scaler.fit_transform(df[existing_numeric])
        print(f"Нормализованы столбцы: {existing_numeric}")
        print(df[existing_numeric].head())
    else:
        print("Нет числовых столбцов для нормализации.")

    print("\n 6: One-Hot Encoding")

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if categorical_cols:
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        print(f"Размер до кодирования: {df.shape}")
        print(f"Размер после кодирования: {df_encoded.shape}")
        print("\nПервые 5 строк итогового датасета:")
        print(df_encoded.head())

        df_final = df_encoded
    else:
        df_final = df


    output_file = "processed_titanic.csv"
    df_final.to_csv(output_file, index=False)
    print(f"\n7: Данные сохранены в файл '{output_file}' ")

