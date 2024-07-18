# predecir_partida.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import re

# Cargar el archivo CSV
def cargar_datos(file_name):
    df = pd.read_csv(file_name, sep=';')
    df.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
    return df

# Función para limpiar valores no numéricos
def limpiar_valor(valor):
    limpio = re.sub(r'[^0-9]', '', valor)
    if limpio == '':
        return np.nan
    else:
        return float(limpio)

# Preprocesar los datos
def preprocesar_datos(df):
    df['TimeAlive'] = df['TimeAlive'].apply(limpiar_valor)
    df['TravelledDistance'] = df['TravelledDistance'].apply(limpiar_valor)
    df = df.dropna(subset=['TimeAlive', 'TravelledDistance'])
    df['TimeAlive'] = df['TimeAlive'].astype(float)
    df['TravelledDistance'] = df['TravelledDistance'].astype(float)
    df['Survived'] = df['Survived'].astype(int)
    df['AbnormalMatch'] = df['AbnormalMatch'].astype(int)
    df = pd.get_dummies(df, columns=['Map', 'Team', 'RoundWinner', 'MatchWinner'], drop_first=True)
    return df

# Entrenar y guardar el modelo
def entrenar_modelo(df):
    X = df[['InternalTeamId', 'MatchId', 'RoundId', 'RLethalGrenadesThrown', 
            'RNonLethalGrenadesThrown', 'PrimaryAssaultRifle', 'PrimarySniperRifle', 
            'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol', 'RoundKills', 
            'RoundAssists', 'RoundHeadshots', 'RoundFlankKills', 
            'RoundStartingEquipmentValue', 'TeamStartingEquipmentValue', 
            'TimeAlive', 'TravelledDistance', 'Survived', 'AbnormalMatch']]
    y = df['MatchWinner_True'].apply(lambda x: 1 if x == True else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'predecir_partida_modelo.pkl')
    print("Modelo guardado como 'predecir_partida_modelo.pkl'")

if __name__ == "__main__":
    file_name = 'base_de_datos_counter_strike.csv'
    df = cargar_datos(file_name)
    df = preprocesar_datos(df)
    entrenar_modelo(df)

# Función para limpiar valores no numéricos
def limpiar_valor(valor):
    limpio = re.sub(r'[^0-9]', '', valor)
    if limpio == '':
        return np.nan
    else:
        return float(limpio)

# Función para preprocesar los datos de un solo jugador
def preprocesar_datos_jugador(data):
    data['TimeAlive'] = limpiar_valor(data['TimeAlive'])
    data['TravelledDistance'] = limpiar_valor(data['TravelledDistance'])
    data['TimeAlive'] = float(data['TimeAlive'])
    data['TravelledDistance'] = float(data['TravelledDistance'])
    data['Survived'] = int(data['Survived'])
    data['AbnormalMatch'] = int(data['AbnormalMatch'])
    data = pd.get_dummies(data, columns=['Map', 'Team', 'RoundWinner', 'MatchWinner'], drop_first=True)
    return data

# Función para predecir la partida con cinco jugadores
def predecir_partida_jugadores(data):
    # Cargar el modelo entrenado
    model = joblib.load('predecir_partida_modelo.pkl')

    # Preprocesar los datos de cada jugador
    for i in range(1, 6):
        jugador_data = {
            'InternalTeamId': data[f'internal_team_id_{i}'],
            'MatchId': data[f'match_id_{i}'],
            'RoundId': data[f'round_id_{i}'],
            'RLethalGrenadesThrown': data[f'rlethal_grenades_thrown_{i}'],
            'RNonLethalGrenadesThrown': data[f'rnonlethal_grenades_thrown_{i}'],
            'PrimaryAssaultRifle': data[f'primary_assault_rifle_{i}'],
            'PrimarySniperRifle': data[f'primary_sniper_rifle_{i}'],
            'PrimaryHeavy': data[f'primary_heavy_{i}'],
            'PrimarySMG': data[f'primary_smg_{i}'],
            'PrimaryPistol': data[f'primary_pistol_{i}'],
            'RoundKills': data[f'round_kills_{i}'],
            'RoundAssists': data[f'round_assists_{i}'],
            'RoundHeadshots': data[f'round_headshots_{i}'],
            'RoundFlankKills': data[f'round_flank_kills_{i}'],
            'RoundStartingEquipmentValue': data[f'round_starting_equipment_value_{i}'],
            'TeamStartingEquipmentValue': data[f'team_starting_equipment_value_{i}'],
            'TimeAlive': data[f'time_alive_{i}'],
            'TravelledDistance': data[f'travelled_distance_{i}'],
            'Survived': data[f'survived_{i}'],
            'AbnormalMatch': data[f'abnormal_match_{i}']
        }
        jugador_df = pd.DataFrame(jugador_data, index=[0])
        jugador_df = preprocesar_datos_jugador(jugador_df)

        # Hacer la predicción para cada jugador
        prediction = model.predict(jugador_df)[0]
        data[f'prediction_{i}'] = 'ganará' if prediction == 1 else 'no ganará'

    return data