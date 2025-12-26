from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Cargar modelos
try:
    print("Cargando modelos...")
    modelo_obesidad = joblib.load('modelo_obesidad.pkl')
    modelo_peso = joblib.load('modelo_peso.pkl')
    metadata = joblib.load('metadata.pkl')
    print("Modelos cargados correctamente")
    print(f"Clasificador: {metadata['best_clf_name']}")
    print(f"Regresor: {metadata['best_reg_name']}")
except Exception as e:
    print(f"Error cargando modelos: {e}")
    modelo_obesidad = None
    modelo_peso = None
    metadata = {'best_clf_name': 'Unknown', 'best_reg_name': 'Unknown', 'scores_clf': {}, 'scores_reg': {}}

# AUXILIARES
def get_imc_category(imc):
    if imc < 18.5: return "bajo peso crítico"
    if imc < 20: return "bajo peso"
    if imc < 25: return "normal"
    if imc < 30: return "sobrepeso"
    if imc < 35: return "obesidad Grado I"
    if imc < 40: return "obesidad Grado II"
    return "obesidad Grado III"

def get_risk_category(proba):
    if proba < 0.25: return "baja"
    if proba < 0.60: return "moderada"
    return "alta"

@app.route('/', methods=['GET'])
def health():
    return jsonify({
        "status": "   API de Salud ML - Activa",
        "version": "2.0",
        "endpoints": {
            "predict": "POST - Generar predicción",
            "health": "GET - Estado del servicio",
            "models": "GET - Información de modelos"
        },
        "modelos_cargados": modelo_obesidad is not None
    })

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "models_disponibles": modelo_obesidad is not None,
        "clasificador": metadata['best_clf_name'],
        "regresor": metadata['best_reg_name']
    })

@app.route('/models')
def model_info():
    return jsonify({
        "clasificacion": {
            "mejor_modelo": metadata['best_clf_name'],
            "metricas": {k: f"{v:.3f}" for k, v in metadata['scores_clf'].items()}
        },
        "regresion": {
            "mejor_modelo": metadata['best_reg_name'],
            "metricas": {k: f"{v:.3f}" for k, v in metadata['scores_reg'].items()}
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if modelo_obesidad is None or modelo_peso is None:
        return jsonify({"error": "Modelos no disponibles"}), 503
    
    try:
        data = request.json
        
        # CAMPOS REQUERIDOS
        required_fields = ['age_years', 'height_cm', 'weight_kg', 'sex', 
                         'ethnicity', 'region', 'activity_level', 
                         'activity_hours_week', 'diet_type', 
                         'pre_existing_cond', 'smoking_status', 
                         'alcohol_units_week']
        
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Campo requerido faltante: {field}"}), 400
        
        # GLUCOSA OPCIONAL
        if 'baseline_glucose_mgdl' not in data or data['baseline_glucose_mgdl'] == '' or data['baseline_glucose_mgdl'] is None:
            data['baseline_glucose_mgdl'] = np.nan
            
        # DATAFRAME
        user_df = pd.DataFrame([{
            'sex': data['sex'],
            'age_years': float(data['age_years']),
            'height_cm': float(data['height_cm']),
            'weight_kg': float(data['weight_kg']),
            'ethnicity': data['ethnicity'],
            'region': data['region'],
            'activity_level': data['activity_level'],
            'activity_hours_week': float(data['activity_hours_week']),
            'diet_type': data['diet_type'],
            'baseline_glucose_mgdl': data['baseline_glucose_mgdl'],
            'pre_existing_cond': data['pre_existing_cond'],
            'smoking_status': data['smoking_status'],
            'alcohol_units_week': float(data['alcohol_units_week'])
        }])

        # CALCULAR IMC ACTUAL
        peso_actual = float(data['weight_kg'])
        estatura_m = float(data['height_cm']) / 100
        imc_actual = peso_actual / (estatura_m ** 2)

        # PREDICCIONES
        proba_obesidad = float(modelo_obesidad.predict_proba(user_df)[0][1])
        delta_peso = float(modelo_peso.predict(user_df)[0])

        # PROYECCIÓN
        years = float(data.get('years', 1))
        peso_final = peso_actual + (delta_peso * years)
        imc_final = peso_final / (estatura_m ** 2)

        # CLASIFICACIONES
        cat_imc_actual = get_imc_category(imc_actual)
        cat_imc_final = get_imc_category(imc_final)
        riesgo = get_risk_category(proba_obesidad)

        # RESPUESTA
        resultado = {
            "nombre": data.get('nombre', 'Usuario'),
            "situacion_actual": {
                "imc": round(imc_actual, 1),
                "categoria_imc": cat_imc_actual,
                "peso_actual": round(peso_actual, 1)
            },
            "prediccion": {
                "probabilidad_obesidad": round(proba_obesidad * 100, 1),
                "categoria_riesgo": riesgo,
                "cambio_peso_anual": round(delta_peso, 1),
                "tipo_cambio": "ganar" if delta_peso >= 0 else "perder"
            },
            "proyeccion": {
                "years": years,
                "peso_final": round(peso_final, 1),
                "imc_final": round(imc_final, 1),
                "categoria_imc_final": cat_imc_final
            },
            "modelos": {
                "clasificador": metadata['best_clf_name'],
                "regresor": metadata['best_reg_name']
            },
            "advertencia": "Este análisis es estadístico y educativo; no constituye un diagnóstico médico."
        }

        return jsonify(resultado)

    except Exception as e:
        return jsonify({"error": f"Error en predicción: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)