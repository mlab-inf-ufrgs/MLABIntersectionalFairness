"""
thesis_datasets.py — Registro das bases usadas nos experimentos da tese
========================================================================

Separado do registro do dashboard (data_module.DATASETS) por dois motivos:
  1. as bases e os resultados da qualificação permanecem como estão;
  2. não depende do Streamlit, então roda em nó de cluster sem interface.

Selecionado com:  python scripts/run_experiments.py --registry thesis ...

Campos de cada base
-------------------
  loader                função sem argumentos que devolve o DataFrame
  target                coluna do desfecho
  favorable_val         valor do desfecho favorável (vira a classe positiva)
  protected_attributes  os dois primeiros formam o par interseccional auditado
                        e NÃO entram como variáveis de entrada dos modelos
  feature_columns       (opcional) lista explícita das variáveis de entrada;
                        se ausente, entram todas as demais colunas
  test_file             (opcional) conjunto de teste oficial, ainda não usado
                        pelo protocolo de validação cruzada

As bases *_v2 e a sintética são geradas por scripts/build_thesis_datasets.py.
"""

import os

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def _parquet(relative_path):
    """Loader de um parquet de data/processed/."""
    def load():
        return pd.read_parquet(os.path.join(PROCESSED_DIR, relative_path))
    return load


def _yearly_parquet(pattern, years):
    """Loader de bases gravadas um arquivo por ano (pattern com {ano}); concatena os anos pedidos.

    Com mais de um ano, acrescenta a coluna `ano` (que não entra como variável de entrada se a
    base tiver feature_columns explícitas).
    """
    def load():
        frames = []
        for year in years:
            df = pd.read_parquet(os.path.join(PROCESSED_DIR, pattern.format(ano=year)))
            if len(years) > 1:
                df.insert(0, "ano", year)
            frames.append(df)
        return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    return load


# --- SINASC: dois recortes de variáveis de entrada sobre o mesmo arquivo --------------
# Pré-natal: informação disponível no início do acompanhamento da gestação.
SINASC_PRENATAL = [
    "uf", "escolaridade_mae", "estado_civil", "tipo_gravidez", "paridade",
    "qtd_gestacoes_anteriores", "qtd_filhos_vivos", "qtd_filhos_mortos",
    "qtd_partos_normais", "qtd_partos_cesareos", "mes_inicio_prenatal",
]
# Nascimento: acrescenta o que só se conhece no parto. O número de consultas e o índice
# de adequação do pré-natal dependem da duração da gestação, por isso ficam neste recorte.
SINASC_NASCIMENTO = SINASC_PRENATAL + [
    "consultas_prenatal_faixa", "consultas_prenatal_n", "adequacao_prenatal",
    "semanas_gestacao", "tipo_parto", "apresentacao", "trabalho_parto_induzido",
    "cesarea_antes_trabalho_parto", "sexo_rn", "local_nascimento", "grupo_robson",
]
# Nunca entram como variável de entrada: idade_mae_anos (atributo protegido em forma
# numérica) e peso_g (define o desfecho).

# O SINASC v2 tem um arquivo por ano (2022-2024). As entradas abaixo usam só 2022, o ano da
# qualificação; o protocolo temporal (treinar em um ano, testar no seguinte) ainda será ligado.
_SINASC_BASE = {
    "loader": _yearly_parquet("thesis/sinasc_v2/sinasc_v2_{ano}.parquet", years=[2022]),
    "target": "desfecho_nascimento",
    "favorable_val": "Normal",
    "protected_attributes": ["raca_cor_mae", "idade_mae"],
    "proxy_attributes": ["escolaridade_mae"],
    "domain": "Saúde materno-infantil (Brasil, 2022)",
}

# --- SIH: um experimento por capítulo da CID-10 (os que a geração da base marcou como incluídos) ---
# Variáveis conhecidas na admissão. A idade em anos entra como ajuste de risco clínico; a faixa
# etária é o 3º atributo protegido (só auditoria) e capitulo_cid é constante dentro do capítulo.
SIH_FEATURES = [
    "uf", "idade_anos", "cid_categoria", "carater_internacao", "complexidade",
    "especialidade_leito", "fora_municipio_residencia", "mes_internacao",
]
SIH_YEAR = 2024
# Capítulos grandes (até 1,4 milhão de internações) usam uma amostra estratificada por
# desfecho x raça/cor x sexo, com semente fixa: mesma amostra em todos os modelos e sementes.
SIH_MAX_ROWS = 200_000
SIH_SAMPLE_SEED = 42


def _sih_chapter(year, chapter, max_rows=SIH_MAX_ROWS):
    def load():
        df = pd.read_parquet(os.path.join(PROCESSED_DIR, "thesis", "sih_v2", str(year),
                                          f"sih_v2_{year}_cap{chapter:02d}.parquet"))
        if max_rows and len(df) > max_rows:
            frac = max_rows / len(df)
            df = (df.groupby(["desfecho", "raca_cor", "sexo"], observed=True, group_keys=False)
                    .sample(frac=frac, random_state=SIH_SAMPLE_SEED)
                    .reset_index(drop=True))
        return df
    return load


def _sih_registry(year=SIH_YEAR):
    path = os.path.join(PROCESSED_DIR, "thesis", "sih_v2", str(year), f"sih_v2_{year}_resumo.parquet")
    if not os.path.exists(path):
        return {}
    resumo = pd.read_parquet(path)
    entries = {}
    for _, row in resumo[resumo["incluido"]].iterrows():
        n = int(row["capitulo_n"])
        entries[f"sih_v2_{year}_cap{n:02d}"] = {
            "loader": _sih_chapter(year, n),
            "target": "desfecho",
            "favorable_val": "Alta",
            "protected_attributes": ["raca_cor", "sexo", "faixa_etaria"],
            "proxy_attributes": [],
            "feature_columns": SIH_FEATURES,
            "n_total": int(row["mantidos"]),
            "domain": f"Internações hospitalares (Brasil, {year}) — {row['capitulo_cid']}",
        }
    return entries


THESIS_DATASETS = {
    "adult": {
        "loader": _parquet("adult_processed.parquet"),
        "target": "income",
        "favorable_val": 1,
        "protected_attributes": ["sex", "race", "age_group"],
        "proxy_attributes": ["education_group", "relationship"],
        "domain": "Renda (EUA, 1994)",
    },
    "acsincome": {
        "loader": _parquet("acsincome_processed.parquet"),
        "target": "PINCP",
        "favorable_val": 1,
        "protected_attributes": ["SEX", "RAC1P", "AGEP_Group"],
        "proxy_attributes": ["SCHL"],
        "domain": "Renda (EUA, ACS 2018, Califórnia)",
    },
    "compas_v2": {
        "loader": _parquet("thesis/compas_v2.parquet"),
        "target": "two_year_recid",
        "favorable_val": 0,
        "protected_attributes": ["sex", "race", "age_group"],
        "proxy_attributes": [],
        "domain": "Justiça criminal (EUA)",
    },
    "sinasc_v2_prenatal": dict(_SINASC_BASE, feature_columns=SINASC_PRENATAL),
    "sinasc_v2_nascimento": dict(_SINASC_BASE, feature_columns=SINASC_NASCIMENTO),
    "intersectional_bias": {
        "loader": _parquet("thesis/intersectional_bias_train.parquet"),
        "test_file": "thesis/intersectional_bias_test.parquet",
        "target": "diagnosis",
        "favorable_val": 0,
        "protected_attributes": ["race", "sex"],
        "proxy_attributes": ["housing"],
        "domain": "Saúde mental (sintética, viés interseccional conhecido)",
    },
    "acspubliccoverage": {
        "loader": _parquet("acspubliccoverage_processed.parquet"),
        "target": "PUBCOV",
        "favorable_val": 1,
        "protected_attributes": ["SEX", "RAC1P", "AGEP_Group"],
        "proxy_attributes": [],
        "domain": "Cobertura pública de saúde (EUA, ACS 2018, Califórnia)",
    },
    **_sih_registry(),
}
