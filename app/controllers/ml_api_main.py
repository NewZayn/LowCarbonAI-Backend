from fastapi import FastAPI, HTTPException
import logging
import traceback
from datetime import datetime
from app.dto.gpu_data import GPUData
from app.dto.prediction_response import Prediction
from app.services.model_service import ModelService
from app.services.data_processing_service import DataProcessingService
from app.services.prediction_service import PredictionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GPU Peak Prediction API")

model_service = ModelService()
data_service = DataProcessingService()
prediction_service = PredictionService()

@app.on_event("startup")
async def startup_event():
    try:
        model_service.load_model()
        logger.info("✅ Modelo carregado com sucesso")
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")

@app.post("/predict", response_model=Prediction)
async def predict(data: GPUData):
    try:
        model_data = model_service.load_model()
        if not model_data:
            raise HTTPException(status_code=500, detail="Modelo não disponível")
        
        if len(data.timestamps) < 100:
            raise HTTPException(
                status_code=400, 
                detail=f"Dados insuficientes: {len(data.timestamps)} pontos"
            )
        
        df_processed, stats = data_service.process_hourly_data(data)
        
        if len(df_processed) < 5:
            raise HTTPException(status_code=400, detail="Dados insuficientes após processamento")
        
        # Gerar previsão
        prediction_result = prediction_service.make_prediction(model_data, df_processed)
        
        logger.info(f"✅ Previsão - GPU {data.gpu_id}: {prediction_result['nextHourAverageUtilization']:.1f}%")
        
        return Prediction(**prediction_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na previsão: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

