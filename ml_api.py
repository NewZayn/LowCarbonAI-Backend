from fastapi import FastAPI, Request
import joblib
import uvicorn
import GPUData
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Any



app = FastAPI(title="GPU Peak Prediction API")
model_data = joblib.load("modelo.pkl")

if isinstance(model_data, dict) and 'model' in model_data:
    model = model_data['model']
    feature_names = model_data['feature_names']
    print(f"Loaded model with {len(feature_names)} features")
else:
    model = model_data
    feature_names = [
        'gpu_utilization (%)',
        'memory_utilization (%)',
        'gpu_power_draw (W)',
        'gpu_temperature (°C)',
        'gpu_fan_speed (%)',
        'gpu_clock_speed (MHz)',
        'cpu_utilization (%)',
        'memory_usage (%)',
        'server_power_draw (W)',
        'server_temperature (°C)',
        'disk_usage (%)',
        'network_bandwidth (Mbps)'
    ]
    print("Loaded legacy model format")

@app.post("/predict")
async def predict(data: GPUData):
    feature_values = [
        data.gpu_utilization__pct,
        data.memory_utilization__pct,
        data.gpu_power_draw__W,
        data.gpu_temperature__C,
        data.gpu_fan_speed__pct,
        data.gpu_clock_speed__MHz,
        data.cpu_utilization__pct,
        data.memory_usage__pct,
        data.server_power_draw__W,
        data.server_temperature__C,
        data.disk_usage__pct,
        data.network_bandwidth__Mbps
    ]
    
    input_df = pd.DataFrame([feature_values], columns=feature_names)
    
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        print("Alerta: Pico detectado!")
    return {"peak": bool(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
