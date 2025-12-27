from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. CARGA DE MODELOS Y METADATOS
# ==========================================
try:
    # Se asume que los archivos .pkl están en la raíz
    modelo_obesidad = joblib.load('modelo_obesidad.pkl')
    modelo_peso = joblib.load('modelo_peso.pkl')
    metadata = joblib.load('metadata.pkl')
    print("Modelos cargados exitosamente.")
except Exception as e:
    print(f"Error cargando modelos: {e}")
    modelo_obesidad = None
    modelo_peso = None
    metadata = {
        'best_clf_name': 'Gradient Boosting', 
        'best_reg_name': 'Gradient Boosting',
        'scores_clf': {}, 'scores_reg': {}
    }

# ==========================================
# 2. FUNCIONES AUXILIARES DE LÓGICA
# ==========================================
def get_imc_category(imc):
    if imc < 16.0: return "bajo peso crítico"
    if imc < 18.5: return "bajo peso"
    if imc < 25.0: return "normal"
    if imc < 30.0: return "sobrepeso"
    if imc < 35.0: return "obesidad Grado I"
    if imc < 40.0: return "obesidad Grado II"
    return "obesidad Grado III"

def get_risk_category(proba):
    if proba < 0.25: return "baja"
    if proba < 0.60: return "moderada"
    return "alta"

# ==========================================
# 3. RUTAS
# ==========================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if modelo_obesidad is None or modelo_peso is None:
        return jsonify({"error": "Los modelos no están disponibles en el servidor"}), 503
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No se recibieron datos"}), 400

        # --- EXTRACCIÓN Y VALIDACIÓN ---
        # Mapeo exacto de los nombres enviados por fetch() en index.html
        age = float(data.get('age'))
        gender = data.get('gender') # 'male' o 'female'
        height = float(data.get('height'))
        weight = float(data.get('weight'))
        region = data.get('region')
        ethnicity = data.get('ethnicity')
        activity_level = data.get('physical_activity_level')
        exercise_hours = float(data.get('exercise_hours_per_week'))
        diet = data.get('dietary_habits')
        conditions = data.get('pre_existing_conditions')
        smoking = data.get('smoking_status')
        alcohol = float(data.get('alcohol_consumption'))
        
        # Glucosa (Opcional en HTML, manejamos valor por defecto si está vacío)
        glucose_raw = data.get('glucose_levels')
        glucose = float(glucose_raw) if glucose_raw and glucose_raw != "" else 95.0
        
        years_projection = float(data.get('years_projection', 1))

        # --- PREPARACIÓN DEL DATAFRAME ---
        # El DataFrame debe tener los nombres de columnas que el modelo vio en el entrenamiento
        input_data = pd.DataFrame([{
            'sex': gender,
            'age_years': age,
            'height_cm': height,
            'weight_kg': weight,
            'ethnicity': ethnicity,
            'region': region,
            'activity_level': activity_level,
            'activity_hours_week': exercise_hours,
            'diet_type': diet,
            'baseline_glucose_mgdl': glucose,
            'pre_existing_cond': conditions,
            'smoking_status': smoking,
            'alcohol_units_week': alcohol
        }])

        # --- CÁLCULOS ACTUALES ---
        height_m = height / 100
        imc_actual = weight / (height_m ** 2)

        # --- PREDICCIONES ---
        # 1. Probabilidad de Obesidad (Clasificación)
        proba_obesidad = float(modelo_obesidad.predict_proba(input_data)[0][1])
        
        # 2. Cambio de peso anual (Regresión)
        delta_peso_anual = float(modelo_peso.predict(input_data)[0])

        # --- PROYECCIÓN ---
        peso_proyectado = weight + (delta_peso_anual * years_projection)
        imc_proyectado = peso_proyectado / (height_m ** 2)

        # --- CATEGORIZACIÓN ---
        categoria_actual = get_imc_category(imc_actual)
        categoria_proyectada = get_imc_category(imc_proyectado)
        nivel_riesgo = get_risk_category(proba_obesidad)

        # --- RESPUESTA JSON ---
        # Los nombres de las llaves coinciden con lo que index.html busca en calculateResults()
        return jsonify({
            "imc_actual": round(imc_actual, 2),
            "categoria_actual": categoria_actual,
            "probabilidad_obesidad": proba_obesidad,
            "nivel_riesgo": nivel_riesgo,
            "cambio_peso_anual": round(delta_peso_anual, 2),
            "peso_proyectado": round(peso_proyectado, 2),
            "imc_proyectado": round(imc_proyectado, 2),
            "categoria_proyectada": categoria_proyectada,
            "modelo_clasificacion": metadata.get('best_clf_name', 'Modelo IA'),
            "modelo_regresion": metadata.get('best_reg_name', 'Modelo IA')
        })

    except Exception as e:
        print(f"Error en el servidor: {str(e)}")
        return jsonify({"error": f"Error procesando la solicitud: {str(e)}"}), 400

# Ruta de salud para Render
@app.route('/health')
def health():
    return jsonify({"status": "ok", "models_loaded": modelo_obesidad is not None})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
