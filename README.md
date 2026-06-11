---
title: Copa 2026 Predict
emoji: ⚽
colorFrom: green
colorTo: yellow
sdk: docker
app_file: app.py
pinned: false
---

# Copa 2026 Predict — Humanos vs. Máquinas

Modelo preditivo para a Copa do Mundo FIFA 2026 usando Regressão de Poisson, Rating ELO e Simulação Monte Carlo.

**[Acesse o app ao vivo]([https://copa2026-predict.streamlit.app](https://copa2026-predict-fyromonubvtaag2awcrucp.streamlit.app/))**

---

## O que é isso?

Um pipeline de Machine Learning que:

1. Aprende com mais de 40 anos de histórico de jogos internacionais (1980–2025)
2. Calcula a força de cada seleção usando Rating ELO dinâmico
3. Prevê gols esperados com Regressão de Poisson
4. Simula o torneio completo 1.000 vezes via Monte Carlo
5. Exibe as probabilidades de título em um dashboard interativo
6. Permite registrar resultados reais e atualizar as previsões em tempo real

A pergunta central: **a máquina consegue prever melhor que um humano?**

---

## Arquitetura

```
data/results.csv (49.450 jogos históricos)
        |
        v
Pipeline Medallion
  Bronze -> Silver -> Pesos -> ELO -> Gold
  Regressão de Poisson (2 modelos GLM)
  Simulação Monte Carlo (N=1000, seed=42)
        |
        v
   Supabase (PostgreSQL)
        |
        v
   Dashboard Streamlit (4 páginas)
```

### Camadas de dados

| Camada | Tabelas | Descrição |
|--------|---------|-----------|
| Bronze | `bronze_jogos` | Dado cru do CSV |
| Silver | `silver_jogos`, `silver_copa2026`, `silver_ponderado`, `silver_elo_pre_jogo`, `silver_elo_atual` | Limpeza, anti-leakage, pesos e ELO |
| Gold | `gold_atributos`, `gold_probabilidades_copa` | Tabela de treino e resultado final |

---

## Metodologia

### Rating ELO

Cada seleção começa em 1500 pontos. A cada jogo, o rating é atualizado com base no resultado e na importância do torneio (amistoso = K20, eliminatórias/competitivos = K40, Copa do Mundo = K60). Jogos mais recentes têm maior peso via decaimento exponencial com meia-vida de 5 anos.

O ELO foi escolhido como única feature do modelo por ser a métrica mais comparável entre confederações — rankings FIFA e forma recente (Copa América, Eurocopa) introduzem vieses por não permitirem participação cruzada entre continentes.

### Regressão de Poisson

Dois modelos GLM Poisson independentes — um para gols do mandante, outro para gols do visitante — treinados com 6 features:

- ELO do time da casa e do visitante
- Diferença de ELO
- Campo neutro (sim/não)
- Peso do torneio
- Peso de recência

Acurácia no holdout temporal (2024+): ~60%, alinhado ao teto esperado para futebol internacional sem dados em tempo real.

### Simulação Monte Carlo

O torneio completo é simulado 1.000 vezes com seed fixo (42) para reprodutibilidade. Em cada simulação, os placares são sorteados das distribuições Poisson estimadas, respeitando a estrutura real da Copa 2026 (grupos A–L, mata-mata de 32 avos). O resultado é a frequência com que cada seleção atingiu cada fase.

### Anti-leakage

Os jogos da Copa 2026 ficam completamente isolados do treinamento. O modelo nunca vê os resultados que está prevendo.

### Atualização em tempo real

A página "Registrar Resultado" permite inserir placares manualmente conforme os jogos acontecem. Ao registrar, o pipeline recalcula o ELO de todas as seleções e roda novamente as 1.000 simulações Monte Carlo, atualizando as probabilidades de título automaticamente.

---

## Dashboard

| Página | Descrição |
|--------|-----------|
| Probabilidades pré-computadas | Top 12 favoritas ao título com gráfico e tabela por fase |
| Simulação ao vivo | Simula um torneio completo aleatório — pódio, grupos e mata-mata |
| Explorador de partidas | Escolha dois times: xG, probabilidades V/E/D, palpite de placar, últimos jogos e H2H |
| Registrar Resultado | Insere placar real e recalcula ELO + Monte Carlo automaticamente |

---

## Resultado do modelo

Top favoritas segundo 1.000 simulações Monte Carlo (pré-torneio):

| Seleção | Prob. Campeã |
|---------|-------------|
| Spain | 16.9% |
| Argentina | 16.4% |
| France | 8.0% |
| Brazil | 6.4% |
| England | 4.7% |

---

## Stack tecnológica

- Python 3.12, pandas, NumPy
- statsmodels — GLM Poisson
- SQLAlchemy + psycopg2 — acesso ao banco
- Supabase (PostgreSQL) — banco de dados na nuvem
- Streamlit + Altair — dashboard interativo
- Streamlit Cloud — deploy público (auto-deploy a cada push no main)

---

## Como rodar localmente

```bash
git clone https://github.com/nasugiyama/copa2026-predict.git
cd copa2026-predict

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # Linux/Mac

pip install -r requirements.txt

cp .env.example .env
# Edite .env e preencha DATABASE_URL com a connection string do Supabase

python src/bronze.py
python src/silver.py
python src/pesos.py
python src/elo.py
python src/gold.py
python src/treino.py
python src/previsao.py
python src/monte_carlo.py

streamlit run app.py
```

---

## Estrutura do projeto

```
copa2026-predict/
├── data/
│   ├── results.csv              # 49.450 jogos históricos (1872–2026)
│   ├── grupos_copa2026.csv      # 48 seleções nos grupos A–L
│   └── calendario_copa2026.csv  # Estrutura do mata-mata (M73–M104)
├── src/
│   ├── db.py          # Conexão com Supabase
│   ├── bronze.py      # Ingestão do CSV
│   ├── silver.py      # Limpeza + anti-leakage
│   ├── pesos.py       # Pesos por torneio e recência
│   ├── elo.py         # Rating ELO dinâmico
│   ├── gold.py        # Tabela de treino final
│   ├── treino.py      # Modelos Poisson
│   ├── previsao.py    # Previsão de partidas
│   ├── monte_carlo.py # Simulação do torneio
│   ├── poisson.py     # Utilitários Poisson
│   └── bandeiras.py   # Bandeiras via flagcdn.com
├── app.py             # Dashboard Streamlit (4 páginas)
└── requirements.txt
```

---

## Referências

- [lvgalvao/IAPredict](https://github.com/lvgalvao/IAPredict) — Arquitetura medallion, pipeline completo e especificações técnicas. Projeto do professor Luciano Galvão, base principal deste trabalho.
- [anesriad/football_WorldCup_2026_predictions](https://github.com/anesriad/football_WorldCup_2026_predictions) — Referência adicional de metodologia e feature engineering com ELO para Copa 2026.
- Dados: [International football results from 1872 to 2024 (Kaggle)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
