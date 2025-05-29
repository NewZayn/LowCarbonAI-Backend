import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Any
import logging
from app.dto.gpu_data import GPUData

logger = logging.getLogger(__name__)

class DataProcessingService:
    """Serviço responsável por processar dados de GPU"""
    
    def process_hourly_data(self, data: GPUData) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Processa 1 hora de dados recebidos da API Java"""
        
        # Validações básicas
        if len(data.timestamps) != len(data.gpu_util):
            raise ValueError("Inconsistência entre timestamps e utilização GPU")
        
        logger.info(f"Dados recebidos: GPU {data.gpu_id} - {len(data.timestamps)} pontos")
        
        # Criar DataFrame base
        df = self._create_base_dataframe(data)
        
        # Adicionar regressores
        df, regressors_added = self._add_regressors(df, data)
        
        # Calcular estatísticas originais
        stats = self._calculate_stats(df, len(data.timestamps))
        
        # Processar dados
        df = self._clean_and_resample(df, regressors_added)
        
        # Adicionar features
        df = self._add_features(df)
        
        logger.info(f"Processamento concluído: {len(df)} pontos finais")
        
        return df, stats
    
    def _create_base_dataframe(self, data: GPUData) -> pd.DataFrame:
        """Cria DataFrame base com timestamps e utilização"""
        dates = [datetime.fromtimestamp(ts/1000) for ts in data.timestamps]
        
        df = pd.DataFrame({
            'ds': dates,
            'y': data.gpu_util
        })
        
        # Tratamento de outliers para GPU utilization
        Q1, Q3 = df['y'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df['y'] = df['y'].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        
        return df
    
    def _add_regressors(self, df: pd.DataFrame, data: GPUData) -> Tuple[pd.DataFrame, list]:
        """Adiciona regressores disponíveis"""
        regressors = []
        
        # Power
        if data.power and len(data.power) == len(data.timestamps):
            power_series = pd.Series(data.power)
            Q1, Q3 = power_series.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            df['gpu_power_draw'] = power_series.clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
            regressors.append('gpu_power_draw')
        
        # Temperature  
        if data.temperature and len(data.temperature) == len(data.timestamps):
            temp_series = pd.Series(data.temperature)
            Q1, Q3 = temp_series.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            df['temperature'] = temp_series.clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
            regressors.append('temperature')
        
        # Memory
        if data.mem_util and len(data.mem_util) == len(data.timestamps):
            mem_series = pd.Series(data.mem_util)
            Q1, Q3 = mem_series.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            df['memory_utilization'] = mem_series.clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
            regressors.append('memory_utilization')
        
        logger.info(f"Regressores adicionados: {regressors}")
        return df, regressors
    
    def _calculate_stats(self, df: pd.DataFrame, original_points: int) -> Dict[str, Any]:
        """Calcula estatísticas dos dados"""
        return {
            'original_points': original_points,
            'processed_points': len(df),
            'data_completeness': len(df) / 720,  # Assumindo 720 pontos esperados
            'avg_utilization': float(df['y'].mean()),
            'max_utilization': float(df['y'].max()),
            'min_utilization': float(df['y'].min()),
            'std_utilization': float(df['y'].std()),
            'time_range': {
                'start': df['ds'].min().isoformat(),
                'end': df['ds'].max().isoformat()
            }
        }
    
    def _clean_and_resample(self, df: pd.DataFrame, regressors: list) -> pd.DataFrame:
        """Limpa e reamostra dados"""
        df.set_index('ds', inplace=True)
        
        # Determinar frequência
        if len(df) > 500:
            freq = '5min'
        elif len(df) > 200:
            freq = '2min'
        else:
            freq = '1min'
        
        # Reamostragem
        agg_dict = {col: 'mean' for col in df.columns}
        df = df.resample(freq).agg(agg_dict).dropna()
        
        # Suavização
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].rolling(window=3, center=True).mean().fillna(df[col])
        
        df.reset_index(inplace=True)
        return df
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features temporais e técnicas"""
        # Features temporais
        df['hour'] = df['ds'].dt.hour
        df['day_of_week'] = df['ds'].dt.dayofweek
        df['is_weekend'] = (df['ds'].dt.dayofweek >= 5).astype(int)
        
        # Features cíclicas
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Features técnicas
        if len(df) >= 5:
            df['trend'] = df['y'].rolling(window=min(5, len(df)), center=True).mean().fillna(df['y'].mean())
            df['volatility'] = df['y'].rolling(window=min(3, len(df))).std().fillna(0)
        
        return df