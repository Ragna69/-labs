import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def main():
    try:
        df = pd.read_csv('train.csv')
        print("Датасет успешно загружен!")
    except FileNotFoundError:
        print(" Ошибка: Файл 'train.csv' не найден.")
        print("Скачайте датасет House Prices с Kaggle и положите train.csv в папку проекта.")
        return

    print("\n ПУНКТ 2: Первые 5 строк датасета ")
    print(df.head())

    print("\n Общая информация о датасете ")
    print(df.info())

    print("\n ПУНКТ 3: Пропуски до обработки ")
    missing_before = df.isnull().sum()
    print(missing_before[missing_before > 0])

    print("\n ПУНКТ 4: Заполнение пропусков ")

    for col in df.columns[df.isnull().any()]:
        if df[col].dtype == 'object':
            # Категориальные признаки -> Мода
            val = df[col].mode()[0]
            df[col].fillna(val, inplace=True)
            print(f"  📝 '{col}' заполнен модой: {val}")
        else:
            # Числовые признаки -> Медиана (устойчива к выбросам)
            val = df[col].median()
            df[col].fillna(val, inplace=True)
            print(f"  📊 '{col}' заполнен медианой: {val}")

    # Доказательство заполнения
    total_missing_after = df.isnull().sum().sum()
    print(f"\n Проверка: осталось пропусков = {total_missing_after} (должно быть 0)")

    print("\n ПУНКТ 5: Нормализация (MinMaxScaler [0,1]) ")

    num_cols = ['LotArea', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'YearBuilt']
    existing_num = [c for c in num_cols if c in df.columns]

    if existing_num:
        scaler = MinMaxScaler()
        df[existing_num] = scaler.fit_transform(df[existing_num])
        print(f"  Нормализованы столбцы: {existing_num}")
        print(df[existing_num].head())
    else:
        print(" Нет числовых столбцов для нормализации")

    print("\n ПУНКТ 6: Кодирование категорий (drop_first=True) ")

    cat_cols = df.select_dtypes(include='object').columns.tolist()

    if cat_cols:
        df_final = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        print(f"  Размер ДО кодирования: {df.shape}")
        print(f"  Размер ПОСЛЕ кодирования: {df_final.shape}")
        print("\n Первые 5 строк итогового датасета:")
        print(df_final.head())
    else:
        df_final = df


if __name__ == "__main__":
    main()