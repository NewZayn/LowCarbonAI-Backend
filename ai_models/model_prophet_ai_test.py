import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import joblib
import logging
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

try:
    logging.info("Carregando dados...")
    df = pd.read_csv("./gpu_data.csv", parse_dates=['timestamp'], index_col='timestamp')
    df = df.asfreq('h')
    
    logging.info(f"Shape do DataFrame: {df.shape}")
    logging.info(f"Colunas: {df.columns.tolist()}")
    logging.info(f"Valores ausentes por coluna:\n{df.isnull().sum()}")
    
    logging.info("Tratando dados ausentes e inválidos...")
    df = df.replace([np.inf, -np.inf], np.nan) 
    df = df.dropna()  
    
    logging.info("Preparando dados para o Prophet...")
    df_prophet = df.reset_index()
    df_prophet = df_prophet.rename(columns={'timestamp': 'ds', 'gpu_utilization (%)': 'y'})
    
    if 'y' not in df_prophet.columns:
        raise ValueError("Coluna 'gpu_utilization (%)' não encontrada nos dados")
    
    if df_prophet['y'].isnull().any() or len(df_prophet['y']) == 0:
        raise ValueError("Valores inválidos na coluna de utilização da GPU")
    
    regressores_adicionados = []
    
    if 'gpu_power_draw (W)' in df.columns:
        df_prophet['gpu_power_draw'] = df['gpu_power_draw (W)'].values
        regressores_adicionados.append('gpu_power_draw')
        logging.info("Regressor 'gpu_power_draw' adicionado")
        
    if 'gpu_temperature (°C)' in df.columns:
        df_prophet['temperature'] = df['gpu_temperature (°C)'].values
        regressores_adicionados.append('temperature')
        logging.info("Regressor 'temperature' adicionado")
        
    if 'memory_utilization (%)' in df.columns:
        df_prophet['memory_utilization'] = df['memory_utilization (%)'].values
        regressores_adicionados.append('memory_utilization')
        logging.info("Regressor 'memory_utilization' adicionado")
    
    train_size = int(len(df_prophet) * 0.8)
    train = df_prophet[:train_size]
    test = df_prophet[train_size:]
    logging.info(f"Dados divididos: {len(train)} amostras para treino, {len(test)} para teste")
    
    logging.info("Criando modelo Prophet...")
    model = Prophet(
        seasonality_mode='multiplicative',  
        changepoint_prior_scale=10,  
        seasonality_prior_scale=100,
        mcmc_samples=0  
    )
    
    for regressor in regressores_adicionados:
        model.add_regressor(regressor)
        logging.info(f"Adicionando regressor '{regressor}' ao modelo")
    
    logging.info("Treinando o modelo...")
    model.fit(train)
    logging.info("Treinamento concluído com sucesso!")
    
    logging.info("Salvando modelo...")
    joblib.dump({'model': model}, 'modelo_regressors.pkl')
    print("Modelo salvo com sucesso em 'modelo_regressors.pkl'")
    
    logging.info("Gerando gráfico de validação...")
    
    future = model.make_future_dataframe(periods=len(test), freq='h')
    
    for regressor in regressores_adicionados:
        if regressor in df_prophet.columns:
            future[regressor] = df_prophet[regressor].values[:len(future)]
    
    forecast = model.predict(future)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(train['ds'], train['y'], label='Treino', color='blue')
    
    plt.plot(test['ds'], test['y'], label='Teste (Real)', color='green')
    
    forecast_test = forecast.iloc[train_size:].reset_index(drop=True)
    plt.plot(test['ds'], forecast_test['yhat'], label='Previsão', color='red', linestyle='--')
    
    plt.fill_between(test['ds'], 
                    forecast_test['yhat_lower'], 
                    forecast_test['yhat_upper'], 
                    color='red', alpha=0.2, label='Intervalo de confiança')
    
    plt.title('Validação do Modelo Prophet - GPU Utilization')
    plt.xlabel('Data')
    plt.ylabel('Utilização da GPU (%)')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('validacao_modelo.png')
    logging.info("Gráfico de validação salvo como 'validacao_modelo.png'")
    
    plt.show()
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    real_values = test['y'].values
    predicted_values = forecast_test['yhat'].values
    
    mae = mean_absolute_error(real_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(real_values, predicted_values))
    
    print(f"Métricas de avaliação:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
except Exception as e:
    logging.error(f"Erro durante o processo: {str(e)}")
    print(f"ERRO: {str(e)}")
