## low-carbon-inference-api (API para geração de previsões)

Esta API tem o objetivo de gerar previsões baseadas em modelos estatísticos treinados, utilizando o Prophet, uma biblioteca para análise e previsão de séries temporais. Ela recebe métricas históricas de uso computacional, processa esses dados e retorna previsões sobre a utilização futura da GPU, auxiliando na redução da pegada de carbono através do monitoramento eficiente dos recursos.

### Pré-requisitos

- Python 3.8 ou superior

---

É recomendado criar um ambiente virtual para gerenciar as dependências:

```bash
python -m venv .venv
source .venv/bin/activate  # No Windows use `.venv\Scripts\activate`
pip install -r requirements.txt


### Execução

A execução da API é realizada através do arquivo `main.py`.

Comando para execução:

```bash
python main.py
```

---

![image](https://github.com/user-attachments/assets/263ca7f1-e725-4ef4-bea9-8a38dede3c8b)


