import pandas as pd
import os

def analyze_datasets(csv_file_path, xlsx_file_path):
    # Verificarea existenței fișierelor
    if not os.path.exists(csv_file_path):
        print(f"Eroare: Fișierul CSV '{csv_file_path}' nu există. Asigură-te că calea și numele fișierului sunt corecte.")
        print("Director curent:", os.getcwd())
        return
    if not os.path.exists(xlsx_file_path):
        print(f"Eroare: Fișierul Excel '{xlsx_file_path}' nu există. Asigură-te că calea și numele fișierului sunt corecte.")
        print("Director curent:", os.getcwd())
        return

    # Citirea setului de date din CSV
    df_csv = pd.read_csv(csv_file_path)

    # Citirea setului de date din Excel
    df_xlsx = pd.read_excel(xlsx_file_path, engine='openpyxl')

    # Verificarea dimensiunilor pentru a ne asigura că datele sunt similare
    csv_shape = df_csv.shape
    xlsx_shape = df_xlsx.shape
    print(f"Dimensiuni fișier CSV: {csv_shape}")
    print(f"Dimensiuni fișier Excel: {xlsx_shape}")

    # Găsirea valorilor lipsă în fișierul CSV
    missing_values_csv = df_csv.isnull().sum()
    print("\nValori lipsă în fișierul CSV:")
    print(missing_values_csv[missing_values_csv > 0])

    # Găsirea valorilor lipsă în fișierul Excel
    missing_values_xlsx = df_xlsx.isnull().sum()
    print("\nValori lipsă în fișierul Excel:")
    print(missing_values_xlsx[missing_values_xlsx > 0])

    # Găsirea rândurilor duplicate în CSV
    duplicate_rows_csv = df_csv[df_csv.duplicated()]
    if not duplicate_rows_csv.empty:
        print("\nRânduri duplicate în fișierul CSV:")
        print(duplicate_rows_csv)
    else:
        print("\nNu există rânduri duplicate în fișierul CSV.")

    # Găsirea rândurilor duplicate în Excel
    duplicate_rows_xlsx = df_xlsx[df_xlsx.duplicated()]
    if not duplicate_rows_xlsx.empty:
        print("\nRânduri duplicate în fișierul Excel:")
        print(duplicate_rows_xlsx)
    else:
        print("\nNu există rânduri duplicate în fișierul Excel.")

if __name__ == "__main__":
    # Definirea căilor către fișierele CSV și Excel
    csv_file_path = './Dataset/cats.csv'
    xlsx_file_path = './Dataset/cats.xlsx'

    # Apelarea funcției de analiză
    analyze_datasets(csv_file_path, xlsx_file_path)
