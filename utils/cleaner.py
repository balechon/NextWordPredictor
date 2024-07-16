import pandas as pd
import numpy as np
import os
from pathlib import Path
import re

def get_the_main_path() -> Path:
    return Path(__file__).resolve().parent

def load_data(name:str) -> pd.DataFrame:
    #if the name input have an extension, raise an error
    if not ('.' in name):
       raise ValueError("The name input should have an extension") 
    
      
    main_path = get_the_main_path()
    file_path = main_path / f'./data/raw/{name}'
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path,encoding='utf-8')


def filter_data(df_in:pd.DataFrame, cols_to_filter: list) -> pd.DataFrame:
    oiginal_columns = df_in.columns
    for col in cols_to_filter:
        if col not in oiginal_columns:
            raise ValueError(f"Column {col} not in DataFrame")
    return df_in[cols_to_filter]

def clean_data(df_in:pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy(deep=True)
    df = df.dropna(thresh=df.shape[0]*0.5, axis=1)
    df = df.dropna(thresh=df.shape[1]*0.5, axis=0)
    df = df.drop_duplicates()
    # filter columns
    cols_to_filter = ['title','transcript'	,'tokens'	,'sentiment','label']
    df = filter_data(df, cols_to_filter)
    df['transcript'] = df['transcript'].apply(cleanner_text)
    return df

def cleanner_text(text:str) -> str:
    # Reemplazar comillas dobles repetidas por comillas simples
    text = text.replace('""', '"')
    # Eliminar paréntesis y su contenido
    text = re.sub(r'\(.*?\)', '', text)
    # Reemplazar dobles guiones por un solo guion
    text = text.replace('--', '-')
    text = re.sub(r'\s+', ' ', text).strip() # Eliminar espacios múltiples
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar instancias de "thank you"
    text = text.replace(' thank you.', '')
    # Eliminar caracteres especiales no deseados, pero mantener puntuación básica
    text = re.sub(r'[^a-z0-9 ,.!?\']', '', text)
    
    return text


def save_data(df:pd.DataFrame, name:str) -> None:
    #if the name input have an extension, raise an error
    if not ('.' in name):
       raise ValueError("The name input should have an extension") 
    main_path = get_the_main_path()
    file_path = main_path / f'./data/final/{name}'
    df.to_csv(file_path, index=False, encoding='utf-8')

def run():
    print("Running cleaner")
    df_raw = load_data('ted_speech.csv')
    df_en = load_data('ted_talks_en.csv')
    df = pd.merge(df_raw, df_en[['title','transcript']], on='title', how='inner')
    df_clean = clean_data(df)
    save_data(df_clean, 'ted_speech_clean.csv')
    print("Cleaner finished correctly")


if __name__ == '__main__':
    run()