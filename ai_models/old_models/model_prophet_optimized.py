import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import joblib
import logging
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def optimize_prophet_params(df_train):
    """Otimiza hiperparâmetros do Prophet com fallback para parâmetros padrão"""
    best_params = None
    best_score = float('inf')
    
    # Parâmetros padrão caso a otimização falhe
    default_params = {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'seasonality_mode': 'additive'
    }
    
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }
    
    logging.info("Iniciando otimização de hiperparâmetros...")
    
    # Verificar se temos dados suficientes para validação cruzada
    if len(df_train) < 100:
        logging.warning("Dados insuficientes para otimização. Usando parâmetros padrão.")
        return default_params, float('inf')
    
    successful_params = []
    
    for changepoint in param_grid['changepoint_prior_scale']:
        for seasonality in param_grid['seasonality_prior_scale']:
            for mode in param_grid['seasonality_mode']:
                try:
                    model = Prophet(
                        changepoint_prior_scale=changepoint,
                        seasonality_prior_scale=seasonality,
                        seasonality_mode=mode,
                        daily_seasonality=True,
                        weekly_seasonality=True,
                        yearly_seasonality=False,
                        mcmc_samples=0
                    )
                    
                    # Adicionar regressores básicos se disponíveis
                    basic_regressors = ['hour', 'day_of_week', 'is_weekend']
                    for reg in basic_regressors:
                        if reg in df_train.columns:
                            model.add_regressor(reg)
                    
                    model.fit(df_train)
                    
                    # Tentar validação cruzada mais conservadora
                    try:
                        df_cv = cross_validation(
                            model, 
                            initial='24 hours', 
                            period='6 hours', 
                            horizon='3 hours',
                            parallel="processes"
                        )
                        df_p = performance_metrics(df_cv)
                        mae = df_p['mae'].mean()
                        
                        successful_params.append({
                            'params': {
                                'changepoint_prior_scale': changepoint,
                                'seasonality_prior_scale': seasonality,
                                'seasonality_mode': mode
                            },
                            'score': mae
                        })
                        
                        if mae < best_score:
                            best_score = mae
                            best_params = {
                                'changepoint_prior_scale': changepoint,
                                'seasonality_prior_scale': seasonality,
                                'seasonality_mode': mode
                            }
                            logging.info(f"Novos melhores parâmetros: {best_params}, MAE: {mae:.4f}")
                    
                    except Exception as cv_error:
                        # Se validação cruzada falhar, usar validação simples
                        logging.debug(f"Validação cruzada falhou para {changepoint}, {seasonality}, {mode}: {cv_error}")
                        
                        # Fazer uma previsão simples para avaliar
                        future = model.make_future_dataframe(periods=10, freq='5T')
                        for reg in basic_regressors:
                            if reg in df_train.columns:
                                future[reg] = df_train[reg].iloc[-1]
                        
                        forecast = model.predict(future)
                        # Score baseado na variabilidade da previsão
                        score = forecast['yhat'].std()
                        
                        successful_params.append({
                            'params': {
                                'changepoint_prior_scale': changepoint,
                                'seasonality_prior_scale': seasonality,
                                'seasonality_mode': mode
                            },
                            'score': score
                        })
                
                except Exception as e:
                    logging.debug(f"Erro com parâmetros {changepoint}, {seasonality}, {mode}: {e}")
                    continue
    
    # Se encontramos parâmetros válidos, usar o melhor
    if successful_params:
        best_result = min(successful_params, key=lambda x: x['score'])
        best_params = best_result['params']
        best_score = best_result['score']
        logging.info(f"Otimização concluída. Melhores parâmetros: {best_params}")
    else:
        # Se nenhuma combinação funcionou, usar parâmetros padrão
        logging.warning("Nenhuma combinação de parâmetros funcionou. Usando parâmetros padrão.")
        best_params = default_params
        best_score = float('inf')
    
    return best_params, best_score

def plot_comparison_graphs(train, test, forecast, train_size, mae, rmse, r2, best_params):
    """Cria gráficos comparativos detalhados"""
    
    # Separar previsões de treino e teste
    train_forecast = forecast.iloc[:train_size]
    test_forecast = forecast.iloc[train_size:]
    
    # Criar figura com subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Gráfico principal - Comparação temporal
    ax1 = plt.subplot(3, 2, (1, 2))
    
    # Plotar dados de treino
    plt.plot(train['ds'], train['y'], 
             label='Dados de Treino', color='blue', alpha=0.7, linewidth=1.5)
    
    # Plotar dados de teste reais
    plt.plot(test['ds'], test['y'], 
             label='Dados de Teste (Real)', color='green', alpha=0.8, linewidth=2)
    
    # Plotar previsões de teste
    plt.plot(test['ds'], test_forecast['yhat'], 
             label='Previsões', color='red', linestyle='--', linewidth=2)
    
    # Intervalo de confiança
    plt.fill_between(test['ds'], 
                     test_forecast['yhat_lower'], 
                     test_forecast['yhat_upper'], 
                     color='red', alpha=0.2, label='Intervalo de Confiança')
    
    plt.title(f'Comparação: Dados Reais vs Previsões\nMAE: {mae:.2f}% | RMSE: {rmse:.2f}% | R²: {r2:.3f}', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Data e Hora', fontsize=12)
    plt.ylabel('Utilização da GPU (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 2. Scatter plot - Real vs Previsto
    ax2 = plt.subplot(3, 2, 3)
    
    plt.scatter(test['y'], test_forecast['yhat'], alpha=0.6, color='purple', s=30)
    
    # Linha de referência perfeita (y=x)
    min_val = min(test['y'].min(), test_forecast['yhat'].min())
    max_val = max(test['y'].max(), test_forecast['yhat'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    
    plt.xlabel('Valores Reais (%)', fontsize=12)
    plt.ylabel('Valores Previstos (%)', fontsize=12)
    plt.title('Correlação: Real vs Previsto', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Adicionar texto com R²
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax2.transAxes, 
             fontsize=12, fontweight='bold', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 3. Análise de resíduos
    ax3 = plt.subplot(3, 2, 4)
    
    residuals = test['y'].values - test_forecast['yhat'].values
    plt.plot(test['ds'], residuals, color='orange', alpha=0.7, linewidth=1.5)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.axhline(y=residuals.std(), color='red', linestyle='--', alpha=0.7, label=f'+1σ ({residuals.std():.2f})')
    plt.axhline(y=-residuals.std(), color='red', linestyle='--', alpha=0.7, label=f'-1σ ({-residuals.std():.2f})')
    
    plt.title('Análise de Resíduos (Erro = Real - Previsto)', fontsize=14, fontweight='bold')
    plt.xlabel('Data e Hora', fontsize=12)
    plt.ylabel('Resíduo (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 4. Distribuição dos erros
    ax4 = plt.subplot(3, 2, 5)
    
    plt.hist(residuals, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    plt.axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Média: {residuals.mean():.2f}%')
    plt.axvline(0, color='green', linestyle='-', linewidth=2, label='Zero (ideal)')
    
    plt.title('Distribuição dos Erros', fontsize=14, fontweight='bold')
    plt.xlabel('Erro (%)', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Métricas e parâmetros
    ax5 = plt.subplot(3, 2, 6)
    ax5.axis('off')
    
    # Criar tabela com métricas
    metrics_text = f"""
    MÉTRICAS DE PERFORMANCE:
    
    • MAE (Erro Médio Absoluto): {mae:.3f}%
    • RMSE (Raiz do Erro Quadrático): {rmse:.3f}%
    • R² (Coeficiente de Determinação): {r2:.3f}
    • Erro Padrão: {residuals.std():.3f}%
    • Erro Médio: {residuals.mean():.3f}%
    
    MELHORES PARÂMETROS:
    
    • Changepoint Prior Scale: {best_params['changepoint_prior_scale']}
    • Seasonality Prior Scale: {best_params['seasonality_prior_scale']}
    • Seasonality Mode: {best_params['seasonality_mode']}
    
    QUALIDADE DO MODELO:
    
    • Acurácia: {'Excelente' if r2 > 0.9 else 'Boa' if r2 > 0.7 else 'Regular' if r2 > 0.5 else 'Ruim'}
    • Pontos de Teste: {len(test)}
    • Pontos de Treino: {len(train)}
    """
    
    ax5.text(0.1, 0.9, metrics_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('modelo_comparacao_detalhada.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfico adicional: Zoom nos últimos dados
    plt.figure(figsize=(15, 8))
    
    # Mostrar apenas últimos 50 pontos para melhor visualização
    n_points = min(50, len(test))
    test_subset = test.tail(n_points)
    forecast_subset = test_forecast.tail(n_points)
    
    plt.plot(test_subset['ds'], test_subset['y'], 
             label='Real', color='green', linewidth=3, marker='o', markersize=4)
    plt.plot(test_subset['ds'], forecast_subset['yhat'], 
             label='Previsto', color='red', linewidth=3, linestyle='--', marker='s', markersize=4)
    
    plt.fill_between(test_subset['ds'], 
                     forecast_subset['yhat_lower'], 
                     forecast_subset['yhat_upper'], 
                     color='red', alpha=0.2, label='Intervalo de Confiança')
    
    plt.title(f'Zoom: Últimos {n_points} Pontos - Real vs Previsto', fontsize=16, fontweight='bold')
    plt.xlabel('Data e Hora', fontsize=12)
    plt.ylabel('Utilização da GPU (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('modelo_zoom_comparacao.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logging.info("Gráficos comparativos salvos: 'modelo_comparacao_detalhada.png' e 'modelo_zoom_comparacao.png'")

try:
    logging.info("Carregando dados...")
    df = pd.read_csv("./server2.csv", parse_dates=['timestamp'], index_col='timestamp')
    
    logging.info(f"Dados originais - Shape: {df.shape}")
    logging.info(f"Colunas disponíveis: {df.columns.tolist()}")
    
    if 'gpu_utilization' not in df.columns:
        raise ValueError("Coluna 'gpu_utilization' não encontrada nos dados")
    
    if len(df) > 2000: 
        logging.info("Detectados dados de alta frequência, agregando para 5 minutos...")
        agg_dict = {}
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                agg_dict[col] = 'mean'
        
        df = df.resample('5T').agg(agg_dict).dropna()
    
    logging.info(f"Dados processados - Shape: {df.shape}")
    logging.info(f"Valores ausentes por coluna:\n{df.isnull().sum()}")
    
    # Tratamento de outliers para todas as colunas numéricas
    logging.info("Tratando outliers...")
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Preparar para Prophet
    df_prophet = df.reset_index()
    df_prophet = df_prophet.rename(columns={'timestamp': 'ds', 'gpu_utilization': 'y'})
    
    # Adicionar regressores baseados nas colunas disponíveis
    regressores_adicionados = []
    
    # Mapear colunas do CSV para nomes padronizados
    regressor_mapping = {
        'gpu_power_draw': 'gpu_power_draw',
        'gpu_temperature': 'temperature', 
        'memory_utilization': 'memory_utilization',
        'cpu_utilization': 'cpu_utilization',
        'gpu_fan_speed': 'fan_speed',
        'gpu_clock_speed': 'clock_speed',
        'server_temperature': 'server_temp'
    }
    
    for original_col, new_col in regressor_mapping.items():
        if original_col in df.columns:
            df_prophet[new_col] = df[original_col].values
            regressores_adicionados.append(new_col)
            logging.info(f"Regressor '{new_col}' adicionado (coluna: {original_col})")
    
    # Features temporais
    df_prophet['hour'] = df_prophet['ds'].dt.hour
    df_prophet['day_of_week'] = df_prophet['ds'].dt.dayofweek
    df_prophet['is_weekend'] = (df_prophet['ds'].dt.dayofweek >= 5).astype(int)
    
    temporal_regressors = ['hour', 'day_of_week', 'is_weekend']
    regressores_adicionados.extend(temporal_regressors)
    
    logging.info(f"Total de regressores: {len(regressores_adicionados)}")
    logging.info(f"Regressores: {regressores_adicionados}")
    
    # Dividir dados
    train_size = int(len(df_prophet) * 0.8)
    train = df_prophet[:train_size]
    test = df_prophet[train_size:]
    
    logging.info(f"Dados divididos: {len(train)} para treino, {len(test)} para teste")
    
    # Otimizar modelo com tratamento de erro
    try:
        best_params, best_score = optimize_prophet_params(train)
        logging.info(f"Melhores parâmetros encontrados: {best_params}")
    except Exception as opt_error:
        logging.error(f"Erro na otimização: {opt_error}")
        # Usar parâmetros padrão
        best_params = {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'seasonality_mode': 'additive'
        }
        logging.info("Usando parâmetros padrão devido a erro na otimização")
    
    # Criar modelo final
    model = Prophet(
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'], 
        seasonality_mode=best_params['seasonality_mode'],
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        mcmc_samples=0
    )
    
    # Adicionar regressores
    for regressor in regressores_adicionados:
        model.add_regressor(regressor)
        logging.info(f"Adicionando regressor '{regressor}' ao modelo")
    
    logging.info("Treinando modelo...")
    model.fit(train)
    
    # Avaliar
    logging.info("Gerando previsões...")
    future = model.make_future_dataframe(periods=len(test), freq='5T')
    
    # Preencher regressores no future dataframe
    for regressor in regressores_adicionados:
        if regressor in df_prophet.columns:
            future[regressor] = df_prophet[regressor].values[:len(future)]
        else:
            future[regressor] = 0
    
    forecast = model.predict(future)
    test_forecast = forecast.iloc[train_size:]
    
    # Calcular métricas
    mae = mean_absolute_error(test['y'], test_forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(test['y'], test_forecast['yhat']))
    r2 = r2_score(test['y'], test_forecast['yhat'])
    
    print(f"\n{'='*60}")
    print(f"MÉTRICAS DO MODELO OTIMIZADO:")
    print(f"{'='*60}")
    print(f"MAE (Erro Médio Absoluto): {mae:.4f}%")
    print(f"RMSE (Raiz do Erro Quadrático): {rmse:.4f}%") 
    print(f"R² (Coeficiente de Determinação): {r2:.4f}")
    print(f"Regressores utilizados: {len(regressores_adicionados)}")
    print(f"{'='*60}")
    
    logging.info("Gerando gráficos comparativos...")
    plot_comparison_graphs(train, test, forecast, train_size, mae, rmse, r2, best_params)
    
    # Salvar modelo otimizado
    model_data = {
        'model': model,
        'regressors': regressores_adicionados,
        'best_params': best_params,
        'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2},
        'data_frequency': '5T',
        'version': '2.0_optimized',
        'column_mapping': regressor_mapping
    }
    
    joblib.dump(model_data, './prophet_model_optimized.pkl')
    logging.info("Modelo otimizado salvo em 'prophet_model_optimized.pkl'!")

    print(f"\nESTATÍSTICAS DOS DADOS:")
    print(f"Período: {df_prophet['ds'].min()} até {df_prophet['ds'].max()}")
    print(f"GPU Utilization - Média: {df_prophet['y'].mean():.2f}% | Desvio: {df_prophet['y'].std():.2f}%")
    print(f"GPU Utilization - Min: {df_prophet['y'].min():.2f}% | Max: {df_prophet['y'].max():.2f}%")
    
except Exception as e:
    logging.error(f"Erro: {str(e)}")
    import traceback
    traceback.print_exc()