from fastapi import FastAPI, HTTPException
import pandas as pd
import uvicorn
from datetime import datetime, timedelta
import  joblib
from app.dto.gpu_data import GPUData
import traceback

app = FastAPI(title="GPU Peak Prediction API")

try:
    model = joblib.load("./ai_models/modelo_regressors.pkl")
    if isinstance(model, dict) and 'model' in model:
        prophet_model = model['model']
    else:
        prophet_model = model
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    prophet_model = None

@app.post("/predict")
async def predict(data: GPUData):
    try:
        if prophet_model is None:
            raise HTTPException(status_code=500, detail="Modelo não carregado")
        
        dates = [datetime.fromtimestamp(ts/1000) for ts in data.timestamps]
        
        df = pd.DataFrame({
            'ds': dates,
            'y': data.gpu_util
        })
        
        regressors_list = []
        if hasattr(prophet_model, 'extra_regressors'):
            for reg in prophet_model.extra_regressors:
                if isinstance(reg, dict):
                    regressors_list.append(reg['name'])
                else:
                    regressors_list.append(reg)
        
        
        if 'gpu_power_draw' in regressors_list and data.power:
            df['gpu_power_draw'] = data.power
        if 'temperature' in regressors_list and data.temperature:
            df['temperature'] = data.temperature
        if 'memory_utilization' in regressors_list and data.mem_util:
            df['memory_utilization'] = data.mem_util
            
        for reg in regressors_list:
            if reg not in df.columns:
                print(f"Regressor '{reg}' não foi fornecido, preenchendo com zeros")
                df[reg] = 0  
                
        future = prophet_model.make_future_dataframe(periods=1, freq='h') 
        
        for reg in regressors_list:
            if reg in df.columns:
                future[reg] = df[reg].iloc[-1] 
            else:
                future[reg] = 0  
                
        print(f"Colunas no dataframe 'future': {future.columns.tolist()}")
        
        forecast = prophet_model.predict(future)
        next_hour_prediction = forecast['yhat'].iloc[-1]
        print(f"Previsão calculada: {next_hour_prediction}")
        
        if pd.isna(next_hour_prediction):
            print("ALERTA: Previsão resultou em valor NaN, retornando 0")
            next_hour_prediction = 0  
        
        return {
            "prediction": float(next_hour_prediction),
            "timestamp": datetime.now().isoformat(),
            "model_version": "Prophet com regressores",
            "input_data_points": len(data.timestamps)
        }
    
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"Erro na previsão: {str(e)}\n{error_detail}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar previsão: {str(e)}")

@app.post("/validate")
async def validate(data: dict):
    """Endpoint para validar os dados recebidos sem processá-los."""
    return {
        "received": data,
        "valid_structure": True
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

