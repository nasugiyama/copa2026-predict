"""Carrega o ranking FIFA histórico e cria silver_fifa_rank_pre_jogo + silver_fifa_rank_atual.

Pré-requisito:
  1. Baixe o dataset "FIFA World Ranking 1992-2024" do Kaggle:
     https://www.kaggle.com/datasets/cashncarry/fifaworldranking
  2. Salve o arquivo CSV em: data/fifa_ranking.csv

Colunas esperadas no CSV: rank_date, country_full, rank, confederation, total_points
(formato do dataset cashncarry/fifaworldranking no Kaggle).

Após rodar este script, execute novamente:
  python src/gold.py
  python src/treino.py
  python src/previsao.py
  python src/monte_carlo.py
"""

from __future__ import annotations

import io
import os

import pandas as pd

from db import get_engine, get_raw_connection

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "fifa_ranking.csv")
MAX_RANK = 220  # teto para normalização (rank 1 = 1.0, rank 220 = ~0.0)

# Mapeamento de nomes entre o dataset FIFA e os nomes usados no projeto.
NOME_MAP = {
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "United States": "United States",
    "China PR": "China PR",
    "Côte d'Ivoire": "Ivory Coast",
    "Cape Verde Islands": "Cape Verde",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Czech Republic": "Czech Republic",
    "DR Congo": "DR Congo",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Republic of Ireland": "Republic of Ireland",
    "Northern Ireland": "Northern Ireland",
    "Palestine": "Palestine",
}

DDL = """
DROP TABLE IF EXISTS silver_fifa_rank_pre_jogo;
CREATE TABLE silver_fifa_rank_pre_jogo (
    id                  bigint generated always as identity primary key,
    jogo_id             bigint,
    rank_casa_norm      double precision,
    rank_visitante_norm double precision
);

DROP TABLE IF EXISTS silver_fifa_rank_atual;
CREATE TABLE silver_fifa_rank_atual (
    id        bigint generated always as identity primary key,
    selecao   text,
    rank_norm double precision
);
"""


def normalizar_rank(rank: float) -> float:
    """Converte posição (1=melhor) em score 0-1 (1.0=melhor)."""
    return max(0.0, 1.0 - (rank - 1) / MAX_RANK)


def carregar_ranking(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["rank_date"])
    df["country_full"] = df["country_full"].str.strip().replace(NOME_MAP)
    df = df.rename(columns={"country_full": "selecao", "rank_date": "data"})
    df = df.sort_values("data")
    return df[["data", "selecao", "rank"]]


def ranking_em_data(ranking: pd.DataFrame, data: pd.Timestamp, selecao: str) -> float:
    """Retorna o ranking normalizado da seleção mais próximo anterior à data."""
    rows = ranking[(ranking["selecao"] == selecao) & (ranking["data"] <= data)]
    if rows.empty:
        return 0.5  # desconhecido → neutro
    return normalizar_rank(float(rows.iloc[-1]["rank"]))


def processar(ranking: pd.DataFrame, eng) -> tuple[pd.DataFrame, pd.DataFrame]:
    jogos = pd.read_sql(
        "SELECT id AS jogo_id, data, time_casa, time_visitante FROM silver_ponderado ORDER BY data, id",
        eng, parse_dates=["data"],
    )
    print(f"  Calculando ranking FIFA para {len(jogos):,} jogos... (pode levar ~2 min)")

    # Pré-agrupar por seleção para lookup eficiente.
    ranking_por_sel: dict[str, pd.DataFrame] = {
        sel: grp.reset_index(drop=True)
        for sel, grp in ranking.groupby("selecao")
    }

    def _rank_norm(selecao: str, data: pd.Timestamp) -> float:
        grp = ranking_por_sel.get(selecao)
        if grp is None:
            return 0.5
        rows = grp[grp["data"] <= data]
        if rows.empty:
            return 0.5
        return normalizar_rank(float(rows.iloc[-1]["rank"]))

    rank_casa = [_rank_norm(r.time_casa, r.data) for r in jogos.itertuples(index=False)]
    rank_visit = [_rank_norm(r.time_visitante, r.data) for r in jogos.itertuples(index=False)]

    pre_jogo = pd.DataFrame({
        "jogo_id": jogos["jogo_id"],
        "rank_casa_norm": rank_casa,
        "rank_visitante_norm": rank_visit,
    })

    # Ranking atual (mais recente de cada seleção).
    data_max = ranking["data"].max()
    atual_rows = []
    for sel, grp in ranking_por_sel.items():
        rows = grp[grp["data"] <= data_max]
        if not rows.empty:
            atual_rows.append({"selecao": sel, "rank_norm": normalizar_rank(float(rows.iloc[-1]["rank"]))})
    atual = pd.DataFrame(atual_rows).sort_values("rank_norm", ascending=False)

    return pre_jogo, atual


def gravar(pre_jogo: pd.DataFrame, atual: pd.DataFrame) -> None:
    pj_buf = io.StringIO()
    pre_jogo.to_csv(pj_buf, index=False, header=False)
    pj_buf.seek(0)

    at_buf = io.StringIO()
    atual.to_csv(at_buf, index=False, header=False)
    at_buf.seek(0)

    conn = get_raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(DDL)
            cur.copy_expert(
                "COPY silver_fifa_rank_pre_jogo (jogo_id, rank_casa_norm, rank_visitante_norm) "
                "FROM STDIN WITH (FORMAT csv, NULL '')", pj_buf,
            )
            cur.copy_expert(
                "COPY silver_fifa_rank_atual (selecao, rank_norm) FROM STDIN WITH (FORMAT csv, NULL '')",
                at_buf,
            )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    if not os.path.exists(CSV_PATH):
        print(f"ERRO: arquivo nao encontrado: {CSV_PATH}")
        print()
        print("Instrucoes:")
        print("  1. Acesse: https://www.kaggle.com/datasets/cashncarry/fifaworldranking")
        print("  2. Clique em 'Download' e extraia o CSV")
        print("  3. Renomeie o arquivo para 'fifa_ranking.csv'")
        print("  4. Coloque em: data/fifa_ranking.csv")
        print("  5. Rode este script novamente")
        return

    eng = get_engine()
    print(f"Lendo {CSV_PATH}...")
    ranking = carregar_ranking(CSV_PATH)
    print(f"  {len(ranking):,} entradas | periodo: {ranking['data'].min().date()} -> {ranking['data'].max().date()}")
    print(f"  selecoes no dataset: {ranking['selecao'].nunique()}")

    pre_jogo, atual = processar(ranking, eng)
    gravar(pre_jogo, atual)

    print("\n" + "=" * 60)
    print("RELATÓRIO RANKING FIFA")
    print("=" * 60)
    print(f"  silver_fifa_rank_pre_jogo: {len(pre_jogo):,} jogos")
    print(f"  silver_fifa_rank_atual:    {len(atual):,} selecoes")
    print("  Top 10 ranking atual (normalizado):")
    for _, r in atual.head(10).iterrows():
        print(f"    {r['selecao']:<25} {r['rank_norm']:.3f}")
    print("=" * 60)
    print("\n[OK] Ranking FIFA carregado.")
    print("\nProximo passo: rode o pipeline novamente:")
    print("  python src/gold.py && python src/treino.py && python src/previsao.py && python src/monte_carlo.py")


if __name__ == "__main__":
    main()
