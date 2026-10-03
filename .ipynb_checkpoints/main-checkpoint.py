import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 
# 1. Загрузка данных с Kaggle (House Prices)
# 
try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Ошибка: Файл 'train.csv' не найден. Скачайте датасет House Prices с Kaggle.")
    df = pd.DataFrame()

if not df.empty:
    print(" Первые 5 строк датасета ")
    print(df.head())

    print("\n Общая информация о датасете ")
    print(df.info())

    print("\n Шаг 3: Количество пропущенных значений ")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    print("\n Шаг 4: Заполнение пропусков ")

    cols_with_missing = df.columns[df.isnull().any()].tolist()

    for col in cols_with_missing:
        if df[col].dtype == 'object':
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
        else:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    print("\nПроверка после заполнения (сумма пропусков должна быть 0):")
    print(df.isnull().sum().sum())

    print("\n Шаг 5: Нормализация данных (MinMaxScaler) ")

    numeric_cols = ['LotArea', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'YearBuilt']

    existing_numeric = [col for col in numeric_cols if col in df.columns]

    if existing_numeric:
        scaler = MinMaxScaler()
        df[existing_numeric] = scaler.fit_transform(df[existing_numeric])
        print(f"Нормализованы столбцы: {existing_numeric}")
        print(df[existing_numeric].head())
    else:
        print("Нет указанных числовых столбцов для нормализации.")

    print("\n Шаг 6: One-Hot Encoding категориальных данных ")

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if categorical_cols:
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        print(f"Размер датасета до кодирования: {df.shape}")
        print(f"Размер датасета после кодирования: {df_encoded.shape}")
        print("\nПервые 5 строк итогового датасета:")
        print(df_encoded.head())

        df_final = df_encoded
    else:
        df_final = df


    output_file = "processed_house_prices.csv"
    df_final.to_csv(output_file, index=False)
    print(f"\n Шаг 7: Данные сохранены в файл '{output_file}' ")
