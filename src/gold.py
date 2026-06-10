"""Spec 05 — Atributos Gold: a tabela de treino ``gold_atributos``.

Junta ``silver_ponderado`` (pesos + alvos) com ``silver_elo_pre_jogo`` (força pré-jogo), deriva
``dif_elo`` e ``forma_recente`` (média dos últimos 5 jogos de cada seleção) e mantém apenas
jogos competitivos. Uma linha por jogo, pronta para o Poisson.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from db import get_engine, get_raw_connection

COLS_ATRIBUTO = [
    "elo_casa", "elo_visitante", "dif_elo", "neutro",
    "peso_torneio", "peso_recencia",
    "forma_casa", "forma_visitante",
    "gols_casa", "gols_visitante",
]

COLUNAS = ["jogo_id", "data", "time_casa", "time_visitante"] + COLS_ATRIBUTO

# Todos os jogos (incluindo amistosos) — para calcular a forma cronologicamente.
QUERY_TODOS = """
SELECT id, data, time_casa, time_visitante, gols_casa, gols_visitante
FROM silver_ponderado
ORDER BY data, id
"""

# Só jogos competitivos, com ELO pré-jogo.
QUERY = """
SELECT s.id AS jogo_id, s.data, s.time_casa, s.time_visitante,
       e.elo_casa, e.elo_visitante,
       s.neutro, s.peso_torneio, s.peso_recencia, s.gols_casa, s.gols_visitante
FROM silver_ponderado s
JOIN silver_elo_pre_jogo e ON e.jogo_id = s.id
WHERE NOT s.eh_amistoso
ORDER BY s.data, s.id
"""

DDL = """
DROP TABLE IF EXISTS gold_atributos;
CREATE TABLE gold_atributos (
    id              bigint generated always as identity primary key,
    jogo_id         bigint,
    data            date,
    time_casa       text,
    time_visitante  text,
    elo_casa        double precision,
    elo_visitante   double precision,
    dif_elo         double precision,
    neutro          boolean,
    peso_torneio    integer,
    peso_recencia   double precision,
    forma_casa      double precision,
    forma_visitante double precision,
    gols_casa       integer,
    gols_visitante  integer
);

DROP TABLE IF EXISTS silver_forma_atual;
CREATE TABLE silver_forma_atual (
    id      bigint generated always as identity primary key,
    selecao text,
    forma   double precision
);
"""


def calcular_forma(todos: pd.DataFrame, n: int = 5) -> tuple[dict, dict]:
    """Calcula forma recente (últimos n jogos) de cada seleção.

    Processa todos os jogos em ordem cronológica (sem leakage).
    Retorna (formas_por_jogo_id, forma_atual_por_selecao).
    """
    historico: dict[str, list[float]] = {}
    formas: dict[int, tuple[float, float]] = {}

    for jogo in todos.itertuples(index=False):
        hist_c = historico.get(jogo.time_casa, [])
        hist_v = historico.get(jogo.time_visitante, [])

        fc = float(np.mean(hist_c[-n:])) if len(hist_c) >= 3 else 0.5
        fv = float(np.mean(hist_v[-n:])) if len(hist_v) >= 3 else 0.5
        formas[jogo.id] = (fc, fv)

        if jogo.gols_casa > jogo.gols_visitante:
            rc, rv = 1.0, 0.0
        elif jogo.gols_casa == jogo.gols_visitante:
            rc, rv = 0.5, 0.5
        else:
            rc, rv = 0.0, 1.0

        historico.setdefault(jogo.time_casa, []).append(rc)
        historico.setdefault(jogo.time_visitante, []).append(rv)

    forma_atual = {
        t: float(np.mean(rs[-n:])) if len(rs) >= 3 else 0.5
        for t, rs in historico.items()
    }
    return formas, forma_atual


def montar(df: pd.DataFrame, formas: dict) -> pd.DataFrame:
    df = df.copy()
    df["dif_elo"] = df["elo_casa"] - df["elo_visitante"]
    df["forma_casa"] = df["jogo_id"].map(lambda jid: formas.get(jid, (0.5, 0.5))[0])
    df["forma_visitante"] = df["jogo_id"].map(lambda jid: formas.get(jid, (0.5, 0.5))[1])

    nulos = df[COLS_ATRIBUTO].isna().sum().sum()
    assert nulos == 0, f"gold_atributos teria {nulos} nulos em colunas de atributo"

    return df[COLUNAS]


def gravar(df: pd.DataFrame, forma_atual: dict) -> int:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, na_rep="")
    buffer.seek(0)

    fa_df = pd.DataFrame(
        sorted(forma_atual.items(), key=lambda kv: kv[1], reverse=True),
        columns=["selecao", "forma"],
    )
    fa_buf = io.StringIO()
    fa_df.to_csv(fa_buf, index=False, header=False, na_rep="")
    fa_buf.seek(0)

    conn = get_raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(DDL)
            cur.copy_expert(
                f"COPY gold_atributos ({', '.join(COLUNAS)}) FROM STDIN WITH (FORMAT csv, NULL '')",
                buffer,
            )
            cur.copy_expert(
                "COPY silver_forma_atual (selecao, forma) FROM STDIN WITH (FORMAT csv, NULL '')",
                fa_buf,
            )
            cur.execute("SELECT COUNT(*) FROM gold_atributos;")
            total = cur.fetchone()[0]
        conn.commit()
        return total
    finally:
        conn.close()


def main() -> None:
    eng = get_engine()

    print("Carregando todos os jogos para calcular forma recente...")
    todos = pd.read_sql(QUERY_TODOS, eng, parse_dates=["data"])
    print(f"  {len(todos):,} jogos no total")
    formas, forma_atual = calcular_forma(todos)

    print("Juntando silver_ponderado x silver_elo_pre_jogo (só competitivos)...")
    df = pd.read_sql(QUERY, eng, parse_dates=["data"])
    print(f"  {len(df):,} jogos competitivos")

    gold = montar(df, formas)
    total = gravar(gold, forma_atual)

    print("\n" + "=" * 60)
    print("RELATÓRIO GOLD")
    print("=" * 60)
    print(f"  gold_atributos:      {total:,} linhas")
    print(f"  silver_forma_atual:  {len(forma_atual):,} seleções")
    print(f"  nulos em atributos:  {int(gold[COLS_ATRIBUTO].isna().sum().sum())}")
    print(f"  período: {gold['data'].min().date()} -> {gold['data'].max().date()}")
    print(f"  forma média casa: {gold['forma_casa'].mean():.3f}  visitante: {gold['forma_visitante'].mean():.3f}")
    print("  Top 5 forma atual:")
    for sel, frm in sorted(forma_atual.items(), key=lambda kv: kv[1], reverse=True)[:5]:
        print(f"    {sel:<22} {frm:.3f}")
    print("=" * 60)
    print("\n[OK] gold_atributos concluída.")


if __name__ == "__main__":
    main()
