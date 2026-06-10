"""Spec 05 — Atributos Gold: a tabela de treino ``gold_atributos``.

Features: ELO pré-jogo, forma recente (últimos 5 jogos), confronto direto H2H
(últimos 10 jogos entre as duas seleções) e, opcionalmente, ranking FIFA normalizado
(requer execução prévia de src/fifa_rank_load.py).
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
    "h2h_casa",
    "gols_casa", "gols_visitante",
]

COLUNAS = ["jogo_id", "data", "time_casa", "time_visitante"] + COLS_ATRIBUTO

QUERY_TODOS = """
SELECT id, data, time_casa, time_visitante, gols_casa, gols_visitante
FROM silver_ponderado
ORDER BY data, id
"""

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
    h2h_casa        double precision,
    gols_casa       integer,
    gols_visitante  integer
);

DROP TABLE IF EXISTS silver_forma_atual;
CREATE TABLE silver_forma_atual (
    id      bigint generated always as identity primary key,
    selecao text,
    forma   double precision
);

DROP TABLE IF EXISTS silver_h2h_atual;
CREATE TABLE silver_h2h_atual (
    id              bigint generated always as identity primary key,
    time_casa       text,
    time_visitante  text,
    h2h             double precision
);
"""

DDL_RANK_COLS = """
ALTER TABLE gold_atributos ADD COLUMN IF NOT EXISTS rank_casa_norm double precision DEFAULT 0.5;
ALTER TABLE gold_atributos ADD COLUMN IF NOT EXISTS rank_visitante_norm double precision DEFAULT 0.5;
"""


def calcular_forma(todos: pd.DataFrame, n: int = 5) -> tuple[dict, dict]:
    """Forma recente: média de resultados (1=vitória, 0.5=empate, 0=derrota) dos últimos n jogos."""
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


def calcular_h2h(todos: pd.DataFrame, n: int = 10) -> tuple[dict, dict]:
    """Confronto direto: aproveitamento do time_casa vs time_visitante nos últimos n encontros."""
    reunioes: dict[frozenset, list[str]] = {}
    h2h_por_jogo: dict[int, float] = {}

    for jogo in todos.itertuples(index=False):
        par = frozenset([jogo.time_casa, jogo.time_visitante])
        hist = reunioes.get(par, [])
        ultimas = hist[-n:]

        if len(ultimas) >= 2:
            pontos = sum(
                1.0 if v == jogo.time_casa else (0.5 if v == "draw" else 0.0)
                for v in ultimas
            )
            h2h_por_jogo[jogo.id] = pontos / len(ultimas)
        else:
            h2h_por_jogo[jogo.id] = 0.5

        if jogo.gols_casa > jogo.gols_visitante:
            vencedor = jogo.time_casa
        elif jogo.gols_casa == jogo.gols_visitante:
            vencedor = "draw"
        else:
            vencedor = jogo.time_visitante
        reunioes.setdefault(par, []).append(vencedor)

    h2h_atual: dict[tuple, float] = {}
    for par, hist in reunioes.items():
        times = list(par)
        t1, t2 = times[0], times[1]
        ultimas = hist[-n:]
        if not ultimas:
            continue
        p1 = sum(1.0 if v == t1 else (0.5 if v == "draw" else 0.0) for v in ultimas)
        taxa = p1 / len(ultimas)
        h2h_atual[(t1, t2)] = taxa
        h2h_atual[(t2, t1)] = 1.0 - taxa

    return h2h_por_jogo, h2h_atual


def _join_fifa_rank(df: pd.DataFrame, eng) -> tuple[pd.DataFrame, bool]:
    """Junta ranking FIFA normalizado se a tabela existir; senão retorna df inalterado."""
    try:
        rank = pd.read_sql(
            "SELECT jogo_id, rank_casa_norm, rank_visitante_norm FROM silver_fifa_rank_pre_jogo", eng
        )
        df = df.merge(rank, on="jogo_id", how="left")
        df["rank_casa_norm"] = df["rank_casa_norm"].fillna(0.5)
        df["rank_visitante_norm"] = df["rank_visitante_norm"].fillna(0.5)
        return df, True
    except Exception:
        return df, False


def montar(df: pd.DataFrame, formas: dict, h2h: dict, usar_fifa_rank: bool = False) -> pd.DataFrame:
    df = df.copy()
    df["dif_elo"] = df["elo_casa"] - df["elo_visitante"]
    df["forma_casa"] = df["jogo_id"].map(lambda jid: formas.get(jid, (0.5, 0.5))[0])
    df["forma_visitante"] = df["jogo_id"].map(lambda jid: formas.get(jid, (0.5, 0.5))[1])
    df["h2h_casa"] = df["jogo_id"].map(lambda jid: h2h.get(jid, 0.5))

    if usar_fifa_rank:
        colunas = ["jogo_id", "data", "time_casa", "time_visitante",
                   "elo_casa", "elo_visitante", "dif_elo", "neutro",
                   "peso_torneio", "peso_recencia",
                   "forma_casa", "forma_visitante", "h2h_casa",
                   "rank_casa_norm", "rank_visitante_norm",
                   "gols_casa", "gols_visitante"]
    else:
        colunas = list(COLUNAS)

    check = [c for c in colunas if c not in ("jogo_id", "data", "time_casa", "time_visitante",
                                               "gols_casa", "gols_visitante")]
    nulos = df[check].isna().sum().sum()
    assert nulos == 0, f"gold_atributos teria {nulos} nulos"
    return df[colunas]


def gravar(df: pd.DataFrame, forma_atual: dict, h2h_atual: dict, usar_fifa_rank: bool) -> int:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, na_rep="")
    buffer.seek(0)

    fa_df = pd.DataFrame(sorted(forma_atual.items(), key=lambda kv: kv[1], reverse=True),
                         columns=["selecao", "forma"])
    fa_buf = io.StringIO()
    fa_df.to_csv(fa_buf, index=False, header=False, na_rep="")
    fa_buf.seek(0)

    h2h_rows = [(t1, t2, v) for (t1, t2), v in h2h_atual.items()]
    h2h_df = pd.DataFrame(h2h_rows, columns=["time_casa", "time_visitante", "h2h"])
    h2h_buf = io.StringIO()
    h2h_df.to_csv(h2h_buf, index=False, header=False, na_rep="")
    h2h_buf.seek(0)

    colunas_gold = list(df.columns)

    conn = get_raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(DDL)
            if usar_fifa_rank:
                cur.execute(DDL_RANK_COLS)
            cur.copy_expert(
                f"COPY gold_atributos ({', '.join(colunas_gold)}) FROM STDIN WITH (FORMAT csv, NULL '')",
                buffer,
            )
            cur.copy_expert(
                "COPY silver_forma_atual (selecao, forma) FROM STDIN WITH (FORMAT csv, NULL '')",
                fa_buf,
            )
            cur.copy_expert(
                "COPY silver_h2h_atual (time_casa, time_visitante, h2h) FROM STDIN WITH (FORMAT csv, NULL '')",
                h2h_buf,
            )
            cur.execute("SELECT COUNT(*) FROM gold_atributos;")
            total = cur.fetchone()[0]
        conn.commit()
        return total
    finally:
        conn.close()


def main() -> None:
    eng = get_engine()

    print("Carregando todos os jogos para forma recente e H2H...")
    todos = pd.read_sql(QUERY_TODOS, eng, parse_dates=["data"])
    print(f"  {len(todos):,} jogos no total")

    formas, forma_atual = calcular_forma(todos)
    h2h_por_jogo, h2h_atual = calcular_h2h(todos)

    print("Juntando silver_ponderado x silver_elo_pre_jogo (só competitivos)...")
    df = pd.read_sql(QUERY, eng, parse_dates=["data"])
    print(f"  {len(df):,} jogos competitivos")

    df, usar_fifa_rank = _join_fifa_rank(df, eng)
    if usar_fifa_rank:
        print("  [OK] ranking FIFA encontrado - adicionado como feature")
    else:
        print("  [--] ranking FIFA nao encontrado - rode fifa_rank_load.py para ativar")

    gold = montar(df, formas, h2h_por_jogo, usar_fifa_rank)
    total = gravar(gold, forma_atual, h2h_atual, usar_fifa_rank)

    print("\n" + "=" * 60)
    print("RELATÓRIO GOLD")
    print("=" * 60)
    print(f"  gold_atributos:      {total:,} linhas")
    print(f"  silver_forma_atual:  {len(forma_atual):,} selecoes")
    print(f"  silver_h2h_atual:    {len(h2h_atual):,} pares ordenados")
    print(f"  features FIFA rank:  {'sim' if usar_fifa_rank else 'nao'}")
    print(f"  periodo: {gold['data'].min().date()} -> {gold['data'].max().date()}")
    print(f"  h2h_casa media: {gold['h2h_casa'].mean():.3f}  (esperado ~0.5)")
    print("=" * 60)
    print("\n[OK] gold_atributos concluída.")


if __name__ == "__main__":
    main()
