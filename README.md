# Copa 2026 Predict

Modelo preditivo para a Copa do Mundo FIFA 2026.

Pipeline: Poisson + ELO + Monte Carlo (1000 simulações)  
Dashboard: Streamlit com 3 páginas interativas

## Setup local

1. Copie `.env.example` para `.env` e preencha `DATABASE_URL`
2. Crie o ambiente virtual: `python -m venv .venv`
3. Ative: `.venv\Scripts\Activate.ps1`
4. Instale as dependências: `pip install -r requirements.txt`
5. Rode o pipeline: `python src/bronze.py` → `silver.py` → `pesos.py` → `elo.py` → `gold.py` → `treino.py` → `previsao.py` → `monte_carlo.py`
6. Rode o app: `streamlit run app.py`
