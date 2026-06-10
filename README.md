---
title: Copa 2026 Predict
emoji: ⚽
colorFrom: green
colorTo: yellow
sdk: docker
app_file: app.py
pinned: false
---

# ⚽ Copa 2026 Predict — Humanos vs. Máquinas

> Modelo preditivo para a Copa do Mundo FIFA 2026 usando Regressão de Poisson, Rating ELO e Simulação Monte Carlo.

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-blue)](https://huggingface.co/spaces/nasugiyama/copa2026-predict)
[![GitHub](https://img.shields.io/badge/GitHub-nasugiyama-black?logo=github)](https://github.com/nasugiyama/copa2026-predict)

---

## 🎯 O que é isso?

Um pipeline de Machine Learning que:
1. **Aprende** com 20 anos de histórico de jogos internacionais (2006–2025)
2. **Calcula** a força de cada seleção usando Rating ELO dinâmico
3. **Prevê** gols esperados com Regressão de Poisson
4. **Simula** o torneio completo 1.000 vezes via Monte Carlo
5. **Exibe** as probabilidades de título em um dashboard interativo

A pergunta central: **a máquina consegue prever melhor que um humano?**

---

## 🏗️ Arquitetura

```
data/results.csv (49.450 jogos históricos)
        │
        ▼
┌─────────────────────────────────────────┐
│           Pipeline Medallion            │
│                                         │
│  Bronze → Silver → Pesos → ELO → Gold  │
│                                         │
│  Regressão de Poisson (2 modelos)       │
│  Simulação Monte Carlo (N=1000, s=42)   │
└─────────────────────────────────────────┘
        │
        ▼
   Supabase (PostgreSQL)
        │
        ▼
   Dashboard Streamlit (3 páginas)
```

### Camadas de dados

| Camada | Tabelas | Descrição |
|--------|---------|-----------|
| **Bronze** | `bronze_jogos` | Dado cru do CSV, renomeado para português |
| **Silver** | `silver_jogos`, `silver_copa2026`, `silver_ponderado`, `silver_elo_pre_jogo`, `silver_elo_atual` | Limpeza, anti-leakage, pesos e ELO |
| **Gold** | `gold_atributos`, `gold_probabilidades_copa` | Tabela de treino e resultado final |

---

## 🤖 Metodologia

### Rating ELO
Cada seleção começa em 1500 pontos. A cada jogo, o rating é atualizado com base no resultado e na importância do torneio (amistoso = K20, eliminatórias = K40, Copa do Mundo = K60). Jogos mais recentes têm maior peso via decaimento exponencial com meia-vida de 5 anos.

### Regressão de Poisson
Dois modelos GLM Poisson independentes (um para gols do time da casa, outro para visitante) são treinados com os seguintes atributos:
- ELO do time da casa e visitante
- Diferença de ELO
- Mando de campo (neutro ou não)
- Peso do torneio e recência

### Simulação Monte Carlo
O torneio completo é simulado 1.000 vezes com seed fixo (42) para reprodutibilidade. Em cada simulação, os placares são sorteados das distribuições Poisson estimadas. O resultado é a frequência com que cada seleção atingiu cada fase.

### Anti-leakage
Os 72 jogos da Copa 2026 ficam **completamente isolados** do treinamento. O modelo nunca "vê" os resultados que está prevendo.

---

## 📊 Dashboard

O app tem 3 páginas:

| Página | Descrição |
|--------|-----------|
| **🏆 Probabilidades pré-computadas** | Top 12 favoritas ao título com gráfico de barras e tabela de probabilidades por fase |
| **🎲 Simulação ao vivo** | Simula um torneio completo aleatoriamente — pódio, grupos e mata-mata |
| **🔍 Explorador de partidas** | Escolha dois times e veja xG esperados e probabilidades V/E/D |

---

## 🚀 Resultado do modelo

Top favoritas segundo 1.000 simulações Monte Carlo:

| Seleção | Prob. Campeã |
|---------|-------------|
| 🇪🇸 Spain | 16.9% |
| 🇦🇷 Argentina | 16.4% |
| 🇫🇷 France | 8.0% |
| 🇧🇷 Brazil | 6.4% |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 4.7% |

---

## 🛠️ Stack tecnológica

- **Python 3.12** — linguagem principal
- **pandas + NumPy** — manipulação de dados
- **statsmodels** — GLM Poisson
- **SciPy** — otimização (matching bipartido para 3os melhores)
- **SQLAlchemy + psycopg2** — acesso ao banco
- **Supabase (PostgreSQL)** — banco de dados na nuvem
- **Streamlit + Altair** — dashboard interativo
- **Hugging Face Spaces** — deploy público

---

## ⚙️ Como rodar localmente

```bash
# 1. Clone o repositório
git clone https://github.com/nasugiyama/copa2026-predict.git
cd copa2026-predict

# 2. Crie e ative o ambiente virtual
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# ou: source .venv/bin/activate  # Linux/Mac

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure o banco de dados
cp .env.example .env
# Edite .env e preencha DATABASE_URL com sua connection string do Supabase

# 5. Rode o pipeline (uma vez)
python src/bronze.py
python src/silver.py
python src/pesos.py
python src/elo.py
python src/gold.py
python src/treino.py
python src/previsao.py
python src/monte_carlo.py

# 6. Suba o dashboard
streamlit run app.py
```

---

## 📁 Estrutura do projeto

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
│   └── bandeiras.py   # Emojis de bandeiras
├── app.py             # Dashboard Streamlit
├── Dockerfile         # Deploy no HF Spaces
└── requirements.txt
```

---

## 👥 Contribuidores

| Nome | Papel |
|------|-------|
| [nasugiyama](https://github.com/nasugiyama) | Autor do projeto |
| [Claude Sonnet 4.6](https://claude.ai) (Anthropic) | Engenharia, pipeline, deploy |

---

## 📚 Referências e inspiração

Este projeto foi desenvolvido com base em:

- **[lvgalvao/IAPredict](https://github.com/lvgalvao/IAPredict)** — Arquitetura medallion, pipeline completo e especificações técnicas. Projeto do professor Luciano Galvão, base principal deste trabalho.
- **[anesriad/football_WorldCup_2026_predictions](https://github.com/anesriad/football_WorldCup_2026_predictions)** — Referência adicional de metodologia e feature engineering com ELO para Copa 2026.
- **Dados:** [International football results (Kaggle)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) — 49.450 jogos internacionais desde 1872.

---

*Feito com dados, Python e 1.000 simulações de Monte Carlo.*
