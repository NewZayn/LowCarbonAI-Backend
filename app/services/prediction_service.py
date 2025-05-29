import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
import logging
from scipy.stats import norm

logger = logging.getLogger(__name__)

class PredictionService:
    
    def make_prediction(self, model_data: Dict[str, Any], df_processed: pd.DataFrame) -> Dict[str, Any]:
        prophet_model = model_data['model']
        regressors = self._get_regressors(model_data, prophet_model)
        future = self._create_future_dataframe(prophet_model, df_processed)
        future = self._fill_regressors(future, df_processed, regressors)
        forecast = prophet_model.predict(future)
        return self._process_results(forecast, df_processed, len(future) - len(df_processed))
    
    def _get_regressors(self, model_data: Dict[str, Any], prophet_model) -> list:
        regressors_from_dict = model_data.get('regressors', [])
        regressors_from_model = []
        if hasattr(prophet_model, 'extra_regressors') and prophet_model.extra_regressors:
            regressors_from_model = list(prophet_model.extra_regressors.keys())
        return regressors_from_model if regressors_from_model else regressors_from_dict
    
    def _create_future_dataframe(self, prophet_model, df_processed: pd.DataFrame):
        if len(df_processed) >= 2:
            time_diff = (df_processed['ds'].iloc[1] - df_processed['ds'].iloc[0]).total_seconds() / 60
            periods_per_hour = int(60 / time_diff)
            future_periods = max(1, periods_per_hour)
            
            freq_map = {1: '1min', 2: '2min', 5: '5min', 10: '10min', 15: '15min', 30: '30min'}
            freq = freq_map.get(int(time_diff), f'{int(time_diff)}min')
        else:
            future_periods = 60
            freq = '1min'
        
        return prophet_model.make_future_dataframe(periods=future_periods, freq=freq)
    
    def _fill_regressors(self, future: pd.DataFrame, df_processed: pd.DataFrame, regressors: list) -> pd.DataFrame:
        for reg in regressors:
            if reg in df_processed.columns:
                if len(df_processed) >= 3:
                    weights = np.exp(np.linspace(-1, 0, min(3, len(df_processed))))
                    weights = weights / weights.sum()
                    last_values = df_processed[reg].tail(len(weights))
                    future_value = np.average(last_values, weights=weights)
                else:
                    future_value = df_processed[reg].iloc[-1]
                
                future[reg] = future_value
            else:
                future_value = self._get_default_regressor_value(reg, df_processed)
                future[reg] = future_value
                logger.warning(f"Regressor '{reg}' não disponível. Usando: {future_value}")  
        return future
    
    def _get_default_regressor_value(self, regressor: str, df_processed: pd.DataFrame) -> float:
        defaults = {
            'cpu_utilization': df_processed['y'].mean() * 0.7,
            'fan_speed': 60.0,
            'clock_speed': 1500.0,
            'server_temp': df_processed.get('temperature', pd.Series([60])).mean() + 5,
            'hour': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'is_weekend': int(datetime.now().weekday() >= 5)
        }
        return defaults.get(regressor, 0.0)
    
    def _process_results(self, forecast: pd.DataFrame, df_processed: pd.DataFrame, future_periods: int) -> Dict[str, Any]:
        future_predictions = forecast.tail(future_periods)
        
        next_hour_avg = future_predictions['yhat'].mean()
        next_hour_max = future_predictions['yhat'].max()
        next_hour_min = future_predictions['yhat'].min()
        
        peak_probability = self._calculate_peak_probability(future_predictions, df_processed)
        
        confidence_width = (future_predictions['yhat_upper'] - future_predictions['yhat_lower']).mean()
        max_range = max(100, df_processed['y'].max() - df_processed['y'].min())
        confidence_score = max(0, 1 - (confidence_width / max_range))
        
        risk_level = self._calculate_risk_level(next_hour_avg, peak_probability * 100)
        trend = self._calculate_trend(next_hour_avg, df_processed)
        
        logger.info(f"Resultado: Avg={next_hour_avg:.1f}%, Pico={peak_probability:.1%}, Risk={risk_level}, Trend={trend}")
        
        return {
            'nextHourAverageUtilization': float(next_hour_avg),
            'nextHourMaximumUtilization': float(next_hour_max),
            'nextHourMinimumUtilization': float(next_hour_min),
            'peakProbabilityPercentage': float(peak_probability * 100),
            'confidenceScore': float(confidence_score),
            'riskLevel': risk_level,
            'trend': trend
        }
    
    def _calculate_peak_probability(self, future_predictions: pd.DataFrame, df_processed: pd.DataFrame) -> float:
        prediction_mean = future_predictions['yhat'].mean()
        
        q75 = df_processed['y'].quantile(0.75)
        q90 = df_processed['y'].quantile(0.90)
        q95 = df_processed['y'].quantile(0.95)
        historical_max = df_processed['y'].max()
        
        if prediction_mean <= q75:
            prob = 0.05
        elif prediction_mean <= q90:
            ratio = (prediction_mean - q75) / (q90 - q75)
            prob = 0.05 + ratio * 0.15
        elif prediction_mean <= q95:
            ratio = (prediction_mean - q90) / (q95 - q90)
            prob = 0.20 + ratio * 0.30
        else:
            ratio = min(1.0, (prediction_mean - q95) / (historical_max - q95))
            prob = 0.50 + ratio * 0.40
        
        if len(df_processed) >= 5:
            recent_trend = df_processed['y'].tail(5).diff().mean()
            trend_factor = min(0.3, max(-0.3, recent_trend / 10))
            prob += trend_factor
        
        return max(0.01, min(0.90, prob))
    
    def _calculate_risk_level(self, prediction_avg: float, peak_probability: float) -> str:
        
        if peak_probability > 60.0:
            risk_from_peak = "HIGH"
        elif peak_probability > 30.0:
            risk_from_peak = "MEDIUM"
        else:
            risk_from_peak = "LOW"
        
        if prediction_avg > 80.0:
            risk_from_util = "HIGH"
        elif prediction_avg > 60.0:
            risk_from_util = "MEDIUM"
        else:
            risk_from_util = "LOW"
        
        risk_levels = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        final_risk = max(risk_from_peak, risk_from_util, key=lambda x: risk_levels[x])
        
        logger.debug(f"Risk calculation: util_avg={prediction_avg:.1f}%, peak_prob={peak_probability:.1f}% -> {final_risk}")
        
        return final_risk
    
    def _calculate_trend(self, prediction_avg: float, df_processed: pd.DataFrame) -> str:        
        historical_window = min(180, len(df_processed))  
        historical_avg = df_processed['y'].tail(historical_window).mean()
        
        if len(df_processed) >= 10:
            recent_trend = df_processed['y'].tail(10).diff().mean()
        else:
            recent_trend = 0
        
        if historical_avg > 0:
            diff_percent = ((prediction_avg - historical_avg) / historical_avg) * 100
        else:
            diff_percent = 0
        
        if diff_percent > 15 or recent_trend > 5:
            trend = "INCREASING"
        elif diff_percent < -15 or recent_trend < -5:
            trend = "DECREASING"
        else:
            trend = "STABLE"
        
        logger.debug(f"Trend calculation: pred={prediction_avg:.1f}%, hist={historical_avg:.1f}%, diff={diff_percent:.1f}%, recent_trend={recent_trend:.2f} -> {trend}")
        
        return trend