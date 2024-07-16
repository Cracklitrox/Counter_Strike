import joblib
from django.shortcuts import render
from django.http import JsonResponse

# Create your views here.

# Cargar el modelo entrenado
# Quitar "#" cuando se quiera entrenar el modelo
# model = joblib.load('paginas/modelo_entrenamiento/modelo_entrenado.pkl')

# Funcion para predecir las kills de la partida del jugador
def predecir_kill(request):
    if request.method == 'POST':
        # Obtener los datos del formulario
        equipo = request.POST.get('Equipo')
        match_id = request.POST.get('MatchId')
        round_id = request.POST.get('RoundId')
        lethal_grenades = request.POST.get('RLethalGrenadesThrown')
        non_lethal_grenades = request.POST.get('RNonLethalGrenadesThrown')
        primary_assault_rifle = request.POST.get('PrimaryAssaultRifle')
        primary_sniper_rifle = request.POST.get('PrimarySniperRifle')
        primary_heavy = request.POST.get('PrimaryHeavy')
        primary_smg = request.POST.get('PrimarySMG')
        primary_pistol = request.POST.get('PrimaryPistol')
        round_kills = request.POST.get('RoundKills')
        round_assists = request.POST.get('RoundAssists')
        round_headshots = request.POST.get('RoundHeadshots')
        round_flank_kills = request.POST.get('RoundFlankKills')
        starting_equipment_value = request.POST.get('RoundStartingEquipmentValue')
        team_starting_equipment_value = request.POST.get('TeamStartingEquipmentValue')

        # Convertir las características a float
        features = [
            float(equipo),
            float(match_id),
            float(round_id),
            float(lethal_grenades),
            float(non_lethal_grenades),
            float(primary_assault_rifle),
            float(primary_sniper_rifle),
            float(primary_heavy),
            float(primary_smg),
            float(primary_pistol),
            float(round_kills),
            float(round_assists),
            float(round_headshots),
            float(round_flank_kills),
            float(starting_equipment_value),
            float(team_starting_equipment_value)
        ]

        # Hacer la predicción
        prediction = model.predict([features])

        # Retornar los resultados en el render
        return render(request, 'predecir_kill.html', {'prediction': prediction[0]})

    return render(request, 'predecir_kill.html')

def index(request):
    return render(request, 'index.html')