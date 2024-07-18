from django.http import JsonResponse
from django.shortcuts import render
import pandas as pd
from .forms import *
import joblib

# Create your views here.

def index(request):
    return render(request, 'index.html')

# Cargar el modelo entrenado
model = joblib.load('paginas/modelo_entrenamiento/predecir_kills_partida.pkl')

# Funcion para predecir las kills de la partida del jugador
def predecir_kill(request):
    if request.method == 'POST':
        # Obtener los datos del formulario
        equipo = float(request.POST.get('Equipo'))
        match_id = float(request.POST.get('MatchId'))
        round_id = float(request.POST.get('RoundId'))
        lethal_grenades = float(request.POST.get('RLethalGrenadesThrown'))
        non_lethal_grenades = float(request.POST.get('RNonLethalGrenadesThrown'))
        primary_assault_rifle = float(request.POST.get('PrimaryAssaultRifle'))
        primary_sniper_rifle = float(request.POST.get('PrimarySniperRifle'))
        primary_heavy = float(request.POST.get('PrimaryHeavy'))
        primary_smg = float(request.POST.get('PrimarySMG'))
        primary_pistol = float(request.POST.get('PrimaryPistol'))
        round_kills = float(request.POST.get('RoundKills'))
        round_assists = float(request.POST.get('RoundAssists'))
        round_headshots = float(request.POST.get('RoundHeadshots'))
        round_flank_kills = float(request.POST.get('RoundFlankKills'))
        starting_equipment_value = float(request.POST.get('RoundStartingEquipmentValue'))
        team_starting_equipment_value = float(request.POST.get('TeamStartingEquipmentValue'))

        # Convertir las características a float
        features = [
            equipo, match_id, round_id, lethal_grenades, non_lethal_grenades,
            primary_assault_rifle, primary_sniper_rifle, primary_heavy, primary_smg,
            primary_pistol, round_kills, round_assists, round_headshots,
            round_flank_kills, starting_equipment_value, team_starting_equipment_value
        ]

        # Hacer la predicción
        prediction = model.predict([features])

        # Retornar los resultados en el render
        return render(request, 'predecir_kill.html', {'resultado': prediction[0]})

    return render(request, 'predecir_kill.html')

model = joblib.load('paginas/modelo_entrenamiento/predecir_ronda_sobrevividas_partida.pkl')
scaler = joblib.load('paginas/modelo_entrenamiento/scaler_ronda_sobrevividas.pkl')

def hacer_prediccion(internal_team_id, match_id, round_id, rlethal_grenades_thrown,
                     rnonlethal_grenades_thrown, primary_assault_rifle, primary_sniper_rifle,
                     primary_heavy, primary_smg, primary_pistol, round_kills, round_assists,
                     round_headshots, round_flank_kills, round_starting_equipment_value,
                     team_starting_equipment_value):
    # Crear un DataFrame con los datos ingresados por el usuario
    user_data = pd.DataFrame({
        'InternalTeamId': [internal_team_id],
        'MatchId': [match_id],
        'RoundId': [round_id],
        'RLethalGrenadesThrown': [rlethal_grenades_thrown],
        'RNonLethalGrenadesThrown': [rnonlethal_grenades_thrown],
        'PrimaryAssaultRifle': [primary_assault_rifle],
        'PrimarySniperRifle': [primary_sniper_rifle],
        'PrimaryHeavy': [primary_heavy],
        'PrimarySMG': [primary_smg],
        'PrimaryPistol': [primary_pistol],
        'RoundKills': [round_kills],
        'RoundAssists': [round_assists],
        'RoundHeadshots': [round_headshots],
        'RoundFlankKills': [round_flank_kills],
        'RoundStartingEquipmentValue': [round_starting_equipment_value],
        'TeamStartingEquipmentValue': [team_starting_equipment_value]
    })

    # Normalizar los datos del usuario usando el mismo scaler
    user_scaled = scaler.transform(user_data)

    # Realizar la predicción
    predicted_match_kills = model.predict(user_scaled)[0]

    return predicted_match_kills

def calcular_promedios(df):
    total_partidas = len(df['MatchId'].unique())
    total_rondas_sobrevividas = 0

    for match_id in df['MatchId'].unique():
        partida_data = df[df['MatchId'] == match_id].iloc[0]
        predicted_kills = hacer_prediccion(
            internal_team_id=partida_data['InternalTeamId'],
            match_id=partida_data['MatchId'],
            round_id=partida_data['RoundId'],
            rlethal_grenades_thrown=partida_data['RLethalGrenadesThrown'],
            rnonlethal_grenades_thrown=partida_data['RNonLethalGrenadesThrown'],
            primary_assault_rifle=partida_data['PrimaryAssaultRifle'],
            primary_sniper_rifle=partida_data['PrimarySniperRifle'],
            primary_heavy=partida_data['PrimaryHeavy'],
            primary_smg=partida_data['PrimarySMG'],
            primary_pistol=partida_data['PrimaryPistol'],
            round_kills=partida_data['RoundKills'],
            round_assists=partida_data['RoundAssists'],
            round_headshots=partida_data['RoundHeadshots'],
            round_flank_kills=partida_data['RoundFlankKills'],
            round_starting_equipment_value=partida_data['RoundStartingEquipmentValue'],
            team_starting_equipment_value=partida_data['TeamStartingEquipmentValue']
        )
        total_rondas_sobrevividas += predicted_kills

    promedio_rondas_sobrevividas = total_rondas_sobrevividas / total_partidas

    promedios_por_mapa = {}
    for mapa in df['Map'].unique():
        mapa_data = df[df['Map'] == mapa]
        total_partidas_mapa = len(mapa_data['MatchId'].unique())
        total_rondas_sobrevividas_mapa = 0

        for match_id in mapa_data['MatchId'].unique():
            partida_data = mapa_data[mapa_data['MatchId'] == match_id].iloc[0]
            predicted_kills = hacer_prediccion(
                internal_team_id=partida_data['InternalTeamId'],
                match_id=partida_data['MatchId'],
                round_id=partida_data['RoundId'],
                rlethal_grenades_thrown=partida_data['RLethalGrenadesThrown'],
                rnonlethal_grenades_thrown=partida_data['RNonLethalGrenadesThrown'],
                primary_assault_rifle=partida_data['PrimaryAssaultRifle'],
                primary_sniper_rifle=partida_data['PrimarySniperRifle'],
                primary_heavy=partida_data['PrimaryHeavy'],
                primary_smg=partida_data['PrimarySMG'],
                primary_pistol=partida_data['PrimaryPistol'],
                round_kills=partida_data['RoundKills'],
                round_assists=partida_data['RoundAssists'],
                round_headshots=partida_data['RoundHeadshots'],
                round_flank_kills=partida_data['RoundFlankKills'],
                round_starting_equipment_value=partida_data['RoundStartingEquipmentValue'],
                team_starting_equipment_value=partida_data['TeamStartingEquipmentValue']
            )
            total_rondas_sobrevividas_mapa += predicted_kills

        if total_partidas_mapa > 0:
            promedio_rondas_sobrevividas_mapa = total_rondas_sobrevividas_mapa / total_partidas_mapa
        else:
            promedio_rondas_sobrevividas_mapa = 0

        promedios_por_mapa[mapa] = promedio_rondas_sobrevividas_mapa

    return promedio_rondas_sobrevividas, promedios_por_mapa

def predecir_rondas_sobrevividas(request):
    if request.method == 'POST':
        df = pd.read_csv('paginas/modelo_entrenamiento/base_de_datos_counter_strike.csv', sep=';')
        
        internal_team_id = int(request.POST['internal_team_id'])
        match_id = int(request.POST['match_id'])
        round_id = int(request.POST['round_id'])
        rlethal_grenades_thrown = int(request.POST['rlethal_grenades_thrown'])
        rnonlethal_grenades_thrown = int(request.POST['rnonlethal_grenades_thrown'])
        primary_assault_rifle = int(request.POST['primary_assault_rifle'])
        primary_sniper_rifle = int(request.POST['primary_sniper_rifle'])
        primary_heavy = int(request.POST['primary_heavy'])
        primary_smg = int(request.POST['primary_smg'])
        primary_pistol = int(request.POST['primary_pistol'])
        round_kills = int(request.POST['round_kills'])
        round_assists = int(request.POST['round_assists'])
        round_headshots = int(request.POST['round_headshots'])
        round_flank_kills = int(request.POST['round_flank_kills'])
        round_starting_equipment_value = int(request.POST['round_starting_equipment_value'])
        team_starting_equipment_value = int(request.POST['team_starting_equipment_value'])

        # Predicción de cuántas rondas sobrevivirá
        predicted_kills = hacer_prediccion(internal_team_id, match_id, round_id, rlethal_grenades_thrown,
                                           rnonlethal_grenades_thrown, primary_assault_rifle, primary_sniper_rifle,
                                           primary_heavy, primary_smg, primary_pistol, round_kills, round_assists,
                                           round_headshots, round_flank_kills, round_starting_equipment_value,
                                           team_starting_equipment_value)

        # Calcular el promedio de rondas sobrevividas por partida
        total_partidas = len(df['MatchId'].unique())
        total_rondas_sobrevividas = 0

        for match_id in df['MatchId'].unique():
            partida_data = df[df['MatchId'] == match_id].iloc[0]
            total_rondas_sobrevividas += hacer_prediccion(partida_data['InternalTeamId'],
                                                          partida_data['MatchId'],
                                                          partida_data['RoundId'],
                                                          partida_data['RLethalGrenadesThrown'],
                                                          partida_data['RNonLethalGrenadesThrown'],
                                                          partida_data['PrimaryAssaultRifle'],
                                                          partida_data['PrimarySniperRifle'],
                                                          partida_data['PrimaryHeavy'],
                                                          partida_data['PrimarySMG'],
                                                          partida_data['PrimaryPistol'],
                                                          partida_data['RoundKills'],
                                                          partida_data['RoundAssists'],
                                                          partida_data['RoundHeadshots'],
                                                          partida_data['RoundFlankKills'],
                                                          partida_data['RoundStartingEquipmentValue'],
                                                          partida_data['TeamStartingEquipmentValue'])

        promedio_rondas_sobrevividas = (total_rondas_sobrevividas / total_partidas) * 100

        # Calcular el promedio de rondas sobrevividas por partida en cada mapa
        mapas = df['Map'].unique()
        promedios_por_mapa = {}

        for mapa in mapas:
            mapa_data = df[df['Map'] == mapa]
            total_partidas_mapa = len(mapa_data['MatchId'].unique())
            total_rondas_sobrevividas_mapa = 0

            for match_id in mapa_data['MatchId'].unique():
                partida_data = mapa_data[mapa_data['MatchId'] == match_id].iloc[0]
                total_rondas_sobrevividas_mapa += hacer_prediccion(partida_data['InternalTeamId'],
                                                                   partida_data['MatchId'],
                                                                   partida_data['RoundId'],
                                                                   partida_data['RLethalGrenadesThrown'],
                                                                   partida_data['RNonLethalGrenadesThrown'],
                                                                   partida_data['PrimaryAssaultRifle'],
                                                                   partida_data['PrimarySniperRifle'],
                                                                   partida_data['PrimaryHeavy'],
                                                                   partida_data['PrimarySMG'],
                                                                   partida_data['PrimaryPistol'],
                                                                   partida_data['RoundKills'],
                                                                   partida_data['RoundAssists'],
                                                                   partida_data['RoundHeadshots'],
                                                                   partida_data['RoundFlankKills'],
                                                                   partida_data['RoundStartingEquipmentValue'],
                                                                   partida_data['TeamStartingEquipmentValue'])

            if total_partidas_mapa > 0:
                promedio_rondas_sobrevividas_mapa = (total_rondas_sobrevividas_mapa / total_partidas_mapa) * 100
            else:
                promedio_rondas_sobrevividas_mapa = 0

            # Convertir el nombre del mapa a formato amigable y almacenar el resultado
            mapa_nombre_amigable = mapa.replace('de_', '').replace('dust2', 'Dust 2').title()
            promedios_por_mapa[mapa_nombre_amigable] = f'{promedio_rondas_sobrevividas_mapa:.1f}% de sobrevivir por ronda'

        response_data = {
            'predicted_kills': f'{predicted_kills:.2f} rondas por partida',
            'promedio_rondas_sobrevividas': f'{promedio_rondas_sobrevividas:.1f}% de sobrevivir por ronda',
            'promedios_por_mapa': promedios_por_mapa
        }

        return JsonResponse(response_data)

    return render(request, 'predecir_rondas_sobrevividas.html')