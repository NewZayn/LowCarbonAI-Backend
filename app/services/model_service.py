import joblib
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelService:
    """Serviço responsável por gerenciar o modelo ML"""
    
    def __init__(self):
        self._model_cache: Optional[Dict[str, Any]] = None
    
    def load_model(self) -> Optional[Dict[str, Any]]:
        """Carrega modelo uma única vez e mantém em cache"""
        if self._model_cache is not None:
            return self._model_cache
        
        try:
            model_path = os.path.join(os.path.dirname(__file__), '../../ai_models/prophet_model_optimized.pkl')
            
            if not os.path.exists(model_path):
                model_path = os.path.join(os.path.dirname(__file__), '../../ai_models/prophet_model.pkl')
                logger.warning("Usando modelo original (não otimizado)")
            
            model_data = joblib.load(model_path)
            
            if isinstance(model_data, dict):
                self._model_cache = model_data
                logger.info(f"Modelo carregado como dict - Versão: {model_data.get('version', '1.0')}")
            else:
                prophet_model = model_data
                regressors = []
                
                if hasattr(prophet_model, 'extra_regressors') and prophet_model.extra_regressors:
                    regressors = list(prophet_model.extra_regressors.keys())
                
                self._model_cache = {
                    'model': prophet_model,
                    'regressors': regressors,
                    'version': '1.0'
                }
            
            logger.info(f"Regressores carregados: {self._model_cache.get('regressors', [])}")
            return self._model_cache
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações do modelo"""
        model_data = self.load_model()
        if not model_data:
            return {"status": "not_loaded"}
        
        return {
            "model_version": model_data.get('version', '1.0'),
            "regressors": model_data.get('regressors', []),
            "performance_metrics": model_data.get('metrics', {}),
            "best_parameters": model_data.get('best_params', {}),
            "data_frequency": model_data.get('data_frequency', 'unknown'),
            "last_loaded": datetime.now().isoformat(),
            "status": "loaded"
        }
    
    def is_loaded(self) -> bool:
        """Verifica se modelo está carregado"""
        return self._model_cache is not None