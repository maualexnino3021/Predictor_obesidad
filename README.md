🥗 Predictor de Riesgo de Obesidad y Evolución de Peso
Desarrollado por: Mauricio Niño Gamboa
Tecnología: Flask (Backend) + Machine Learning + Tailwind CSS (Frontend)
Esta aplicación web es una herramienta avanzada de análisis predictivo que utiliza Inteligencia Artificial para evaluar el estado nutricional actual de un usuario y proyectar su evolución física en un periodo de hasta 5 años.
🚀 Características Principales

* Doble Análisis Predictivo:

Clasificador: Determina el riesgo de padecer obesidad (Binario/Multiclase).
Regresor: Estima la ganancia o pérdida de peso proyectada en kilogramos según los hábitos del usuario.


* Interfaz Moderna: Diseño responsivo construido con Tailwind CSS, utilizando un formulario interactivo por pasos.
* Métricas de Rendimiento: El sistema calcula y muestra en tiempo real la precisión del modelo (F1-Score) y el error medio (MAE) basándose en los modelos cargados.
* Visualización de Datos: Inclusión de una Matriz de Confusión generada dinámicamente para validar la fiabilidad de la predicción.
* Reporte PDF: Capacidad de exportar los resultados y consejos de salud a un documento PDF profesional.

🛠️ Stack Tecnológico

* Backend: Flask (Python 3.x)
* Machine Learning: Scikit-learn, Pandas, NumPy, Joblib.
* Frontend: HTML5, Jinja2 (Motores de plantillas), Tailwind CSS, JavaScript (ES6).
* Servidor de Producción: Gunicorn (listo para despliegue en Render/Heroku).

📂 Estructura del Proyecto
.
├── app.py                  # Servidor Flask y lógica de inferencia
├── modelo_obesidad.pkl     # Modelo entrenado para clasificación (Riesgo)
├── modelo_peso.pkl         # Modelo entrenado para regresión (Tendencia)
├── metadata.pkl            # Métricas y parámetros de entrenamiento
├── requirements.txt        # Librerías necesarias
├── static/                 # Archivos CSS, JS e Imágenes
└── templates/
    └── index.html          # Interfaz de usuario principal

⚙️ Instalación y Configuración
Sigue estos pasos para ejecutar el proyecto localmente:

1. 
Clonar el repositorio:
bashDownloadCopy codegit clone <tu-repositorio-url>
cd <nombre-del-proyecto>

2. 
Crear y activar un entorno virtual:
bashDownloadCopy codepython -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

3. 
Instalar dependencias:
bashDownloadCopy codepip install -r requirements.txt

4. 
Ejecutar la aplicación:
bashDownloadCopy codepython app.py
La aplicación estará disponible en http://127.0.0.1:10000.


📊 Funcionamiento del Modelo
El sistema utiliza archivos .pkl que contienen modelos previamente entrenados (ej. Gradient Boosting o Random Forest).

* Al recibir los datos del formulario (edad, peso, altura, actividad física, etc.), el backend procesa la información y realiza dos consultas simultáneas a los modelos.
* El resultado incluye no solo la predicción, sino también un análisis de Importancia de Características, permitiendo al usuario saber qué hábito influye más en su peso.

📝 Notas de Uso

* Escalabilidad: Al estar basado en Flask, este proyecto es fácilmente integrable con bases de datos SQL (como PostgreSQL) para guardar históricos de consultas.
* Entrenamiento: Si deseas re-entrenar los modelos, asegúrate de actualizar los archivos .pkl en la raíz del proyecto.

⚠️ Descargo de Responsabilidad
Este software es una herramienta de orientación estadística basada en datos. Los resultados no sustituyen un diagnóstico médico profesional. Siempre consulte a un nutricionista o profesional de la salud antes de realizar cambios drásticos en su dieta o estilo de vida.
