import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Leer el archivo CSV
df = pd.read_csv('base_de_datos_counter_strike.csv', sep=';')

# Variables dependientes e independientes
X = df[['InternalTeamId', 'MatchId', 'RoundId', 'RLethalGrenadesThrown', 
        'RNonLethalGrenadesThrown', 'PrimaryAssaultRifle', 'PrimarySniperRifle', 
        'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol', 'RoundKills', 
        'RoundAssists', 'RoundHeadshots', 'RoundFlankKills', 
        'RoundStartingEquipmentValue', 'TeamStartingEquipmentValue']]
y = df['MatchKills']

# Normalización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.6, random_state=42)

# Entrenar el modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar el modelo y el scaler
joblib.dump(model, 'predecir_ronda_sobrevividas_partida.pkl')
joblib.dump(scaler, 'scaler_ronda_sobrevividas.pkl')
print("Modelo y scaler guardados como 'predecir_ronda_sobrevividas_partida.pkl' y 'scaler_ronda_sobrevividas.pkl'")