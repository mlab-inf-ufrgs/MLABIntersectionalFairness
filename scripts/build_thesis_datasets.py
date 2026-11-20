"""
build_thesis_datasets.py — Bases de dados para os experimentos da tese
======================================================================

Gera, em data/processed/thesis/, versões novas das bases SEM alterar os arquivos
de data/processed/ usados na qualificação:

  compas_v2.parquet                  COMPAS com as variáveis numéricas recuperadas
  sinasc_v2/sinasc_v2_{ano}.parquet          SINASC de um ano, com variáveis da Declaração de Nascido Vivo
  sinasc_v2/sinasc_v2_{ano}_ausencia.csv     preenchimento de cada variável bruta, por raça/cor da mãe
  sinasc_v2/sinasc_v2_{ano}_exclusoes.csv    registros excluídos, por motivo
  sih_v2/{ano}/sih_v2_{ano}_cap{NN}.parquet  SIH: internações do ano, um arquivo por capítulo da CID-10
  sih_v2/{ano}/sih_v2_{ano}_resumo.parquet   SIH: AIHs, mantidas e % de óbito por capítulo (lido pelo painel)
  sih_v2/{ano}/sih_v2_{ano}_exclusoes.csv    SIH: funil de exclusões, total e por capítulo
  intersectional_bias_train.parquet  OpenML 44202 (10.000 instâncias)
  intersectional_bias_test.parquet   OpenML 44203 (1.000 instâncias)

Uso
---
  python scripts/build_thesis_datasets.py --only compas synthetic
  python scripts/download_opendatasus.py --only sinasc      # baixa os CSVs nacionais do SINASC
  python scripts/build_thesis_datasets.py --only sinasc     # 2022-2024 a partir dos CSVs
  python scripts/build_thesis_datasets.py --only sinasc --sinasc-years 2022 --sinasc-source dbc
                                                            # FTP do DATASUS; requer pyreaddbc e dbfread
  python scripts/download_datasus_ftp.py --inicio 2024-01 --fim 2025-06
  python scripts/build_thesis_datasets.py --only sih        # internações de 2024; requer pyreaddbc

Cada base é conferida contra a versão da qualificação: aplicadas as mesmas regras,
as colunas em comum devem coincidir.
"""

import argparse
import glob
import os
import sys
import tempfile

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_DIR = os.path.join(PROCESSED_DIR, "thesis")

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ---------------------------------------------------------------------------
# COMPAS
# ---------------------------------------------------------------------------

def build_compas():
    """
    Mesmos filtros de linhas da qualificação (raça, idade 19-65, grau da acusação),
    com variáveis adicionais do arquivo da ProPublica. Não entram: os escores do
    próprio COMPAS (decile_score, score_text, v_*), nem variáveis posteriores ao
    desfecho (is_recid, r_*, vr_*).
    """
    csv_path = os.path.join(PROJECT_ROOT, "compas-scores-two-years.csv")
    df = pd.read_csv(csv_path)

    df = df[df["race"].isin(["Caucasian", "African-American", "Hispanic"])].copy()
    df = df[(df["age"] >= 19) & (df["age"] <= 65)].copy()
    df["age_group"] = pd.cut(df["age"], bins=[18, 35, 50, 65], labels=["Young", "Middle-aged", "Senior"]).astype(str)
    df = df[df["c_charge_degree"].isin(["F", "M"])].copy()
    df["c_charge_degree"] = df["c_charge_degree"].map({"F": "Felony", "M": "Misdemeanor"})

    # Variável da qualificação, mantida apenas para conferência
    priors_group = pd.cut(df["priors_count"], bins=[-1, 0, 3, np.inf], labels=["0", "1-3", "4+"]).astype(str)

    # Dias de prisão provisória associados à acusação (ausente quando não há registro)
    jail_in = pd.to_datetime(df["c_jail_in"], errors="coerce")
    jail_out = pd.to_datetime(df["c_jail_out"], errors="coerce")
    df["days_in_jail"] = (jail_out - jail_in).dt.total_seconds() / 86400.0
    df.loc[df["days_in_jail"] < 0, "days_in_jail"] = np.nan

    out = df[[
        "two_year_recid", "sex", "race", "age_group",
        "age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count",
        "c_charge_degree", "days_in_jail",
    ]].reset_index(drop=True)

    # Conferência com a base da qualificação
    ref = pd.read_parquet(os.path.join(PROCESSED_DIR, "compas_processed.parquet"))
    chk = out[["two_year_recid", "sex", "race", "age_group", "c_charge_degree"]].copy()
    chk["priors_count_group"] = priors_group.reset_index(drop=True)
    chk = chk[ref.columns]
    same = len(chk) == len(ref) and bool((chk.astype(str).values == ref.astype(str).values).all())
    print(f"[COMPAS] v2: {out.shape[0]:,} linhas x {out.shape[1]} colunas")
    print(f"[COMPAS] colunas da qualificação reproduzidas linha a linha: {same}")
    print(f"[COMPAS] days_in_jail ausente: {out['days_in_jail'].isna().sum()} registros")

    path = os.path.join(OUT_DIR, "compas_v2.parquet")
    out.to_parquet(path, index=False, compression="zstd")
    print(f"[COMPAS] salvo em {path}")


# ---------------------------------------------------------------------------
# Base sintética (OpenML): treino e teste oficiais
# ---------------------------------------------------------------------------

def build_synthetic():
    from sklearn.datasets import fetch_openml

    for data_id, split in ((44202, "train"), (44203, "test")):
        df = fetch_openml(data_id=data_id, as_frame=True, parser="auto").frame
        df.columns = df.columns.str.lower()
        df["diagnosis"] = df["diagnosis"].astype(int)
        path = os.path.join(OUT_DIR, f"intersectional_bias_{split}.parquet")
        df.to_parquet(path, index=False, compression="zstd")
        print(f"[SINTÉTICA] OpenML {data_id} ({split}): {df.shape[0]:,} linhas x {df.shape[1]} colunas -> {path}")

    legacy = pd.read_parquet(os.path.join(PROCESSED_DIR, "intersectional_bias_processed.parquet"))
    test = pd.read_parquet(os.path.join(OUT_DIR, "intersectional_bias_test.parquet"))
    same = legacy.shape == test.shape and bool(
        (legacy[test.columns].astype(str).values == test.astype(str).values).all())
    print(f"[SINTÉTICA] intersectional_bias_processed.parquet (qualificação) é o conjunto de TESTE: {same}")


# ---------------------------------------------------------------------------
# SINASC
# ---------------------------------------------------------------------------

UFS = ['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA',
       'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN',
       'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO']

# Código IBGE da UF (2 primeiros dígitos do município) -> sigla
UF_IBGE = {'11': 'RO', '12': 'AC', '13': 'AM', '14': 'RR', '15': 'PA', '16': 'AP', '17': 'TO',
           '21': 'MA', '22': 'PI', '23': 'CE', '24': 'RN', '25': 'PB', '26': 'PE', '27': 'AL',
           '28': 'SE', '29': 'BA', '31': 'MG', '32': 'ES', '33': 'RJ', '35': 'SP', '41': 'PR',
           '42': 'SC', '43': 'RS', '50': 'MS', '51': 'MT', '52': 'GO', '53': 'DF'}

# Rótulos conferidos no dicionário oficial "SINASC: Estrutura de 1996 a 2019" (CGIAE/SVS/MS),
# baixado do Portal de Dados Abertos do SUS (data/raw/opendatasus/sinasc/dicionario_sinasc.pdf).
RACA_MAP = {'1': 'Branca', '2': 'Preta', '3': 'Amarela', '4': 'Parda', '5': 'Indígena'}
# ESCMAE é medida em anos de estudo concluídos, não em nível de ensino.
ESC_MAP = {'1': 'Nenhuma', '2': '1 a 3 anos', '3': '4 a 7 anos', '4': '8 a 11 anos', '5': '12 anos ou mais'}

# Variáveis categóricas adicionais: coluna bruta -> (nome novo, rótulos conforme o dicionário do SINASC).
# Códigos fora do dicionário (9 = ignorado, vazio) viram a categoria "Ignorado".
# KOTELCHUCK: o dicionário não lista os códigos; os rótulos seguem as categorias do índice de
# Kotelchuck usadas no TabNet/DATASUS (1 a 5; 9 = ignorado).
# TPROBSON: grupos 1 a 10 de Robson; o código 11 (gerado pelo sistema) indica registro sem
# informação suficiente para classificar.
CATEGORICAS = {
    'ESTCIVMAE': ('estado_civil', {'1': 'Solteira', '2': 'Casada', '3': 'Viúva',
                                   '4': 'Separada/Divorciada', '5': 'União estável'}),
    'GRAVIDEZ': ('tipo_gravidez', {'1': 'Única', '2': 'Dupla', '3': 'Tripla ou mais'}),
    'PARIDADE': ('paridade', {'0': 'Nulípara', '1': 'Multípara'}),
    'CONSULTAS': ('consultas_prenatal_faixa', {'1': 'Nenhuma', '2': '1 a 3', '3': '4 a 6', '4': '7 ou mais'}),
    'KOTELCHUCK': ('adequacao_prenatal', {'1': 'Não fez pré-natal', '2': 'Inadequado', '3': 'Intermediário',
                                          '4': 'Adequado', '5': 'Mais que adequado'}),
    'PARTO': ('tipo_parto', {'1': 'Vaginal', '2': 'Cesáreo'}),
    'TPAPRESENT': ('apresentacao', {'1': 'Cefálica', '2': 'Pélvica ou podálica', '3': 'Transversa'}),
    'STTRABPART': ('trabalho_parto_induzido', {'1': 'Sim', '2': 'Não', '3': 'Não se aplica'}),
    'STCESPARTO': ('cesarea_antes_trabalho_parto', {'1': 'Sim', '2': 'Não', '3': 'Não se aplica'}),
    'SEXO': ('sexo_rn', {'1': 'Masculino', '2': 'Feminino', 'M': 'Masculino', 'F': 'Feminino'}),
    'LOCNASC': ('local_nascimento', {'1': 'Hospital', '2': 'Outro estabelecimento de saúde',
                                     '3': 'Domicílio', '4': 'Outros', '5': 'Aldeia indígena'}),
    'TPROBSON': ('grupo_robson', {**{f'{i:02d}': f'Grupo {i}' for i in range(1, 11)},
                                  '11': 'Não classificado'}),
}

# Variáveis numéricas adicionais: coluna bruta -> (nome novo, código de "ignorado")
NUMERICAS = {
    'QTDGESTANT': ('qtd_gestacoes_anteriores', 99),
    'QTDFILVIVO': ('qtd_filhos_vivos', 99),
    'QTDFILMORT': ('qtd_filhos_mortos', 99),
    'QTDPARTNOR': ('qtd_partos_normais', 99),
    'QTDPARTCES': ('qtd_partos_cesareos', 99),
    'MESPRENAT': ('mes_inicio_prenatal', 99),
    'CONSPRENAT': ('consultas_prenatal_n', 99),
    'SEMAGESTAC': ('semanas_gestacao', 99),
}

SINASC_RAW_COLS = (['CODMUNRES', 'RACACORMAE', 'IDADEMAE', 'ESCMAE', 'PESO']
                   + list(CATEGORICAS) + list(NUMERICAS))
OPENDATASUS_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "opendatasus")


def _clean(series):
    """Texto sem espaços; vazio/None viram NaN."""
    s = series.astype(str).str.strip()
    return s.where(~s.isin(['', 'None', 'nan', 'NaN']), np.nan)


def _read_dbc(dbc_path, tmp_dir):
    import pyreaddbc
    from dbfread import DBF

    dbf_path = os.path.join(tmp_dir, os.path.basename(dbc_path).replace('.dbc', '.dbf'))
    pyreaddbc.dbc2dbf(dbc_path, dbf_path)
    df = pd.DataFrame(iter(DBF(dbf_path, encoding='latin-1')))
    os.remove(dbf_path)
    df.columns = [c.upper() for c in df.columns]
    return df


def _process_sinasc_uf(raw, uf):
    """Devolve (base decodificada com todas as linhas, máscaras de exclusão, ausência por variável)."""
    res = pd.DataFrame(index=raw.index)
    res['uf'] = uf

    raca = _clean(raw['RACACORMAE']).map(RACA_MAP)
    idade = pd.to_numeric(_clean(raw['IDADEMAE']), errors='coerce')
    esc = _clean(raw['ESCMAE']).map(ESC_MAP)
    peso = pd.to_numeric(_clean(raw['PESO']), errors='coerce')

    # --- Regras da qualificação (para conferência) ---
    idade_grp_v1 = pd.cut(idade, bins=[0, 19, 34, 100], labels=['Jovem (<20)', 'Adulta (20-34)', 'Madura (>34)'])
    keep_v1 = raca.notna() & idade_grp_v1.notna() & esc.notna()
    desfecho_v1 = np.where(peso < 2500, 'Baixo Peso', 'Normal')  # peso ausente contava como Normal

    # --- Regras da v2 ---
    idade_v2 = idade.where((idade >= 10) & (idade <= 65))       # 99 = idade ignorada
    peso_v2 = peso.where((peso >= 100) & (peso < 9000))         # 9999 = peso ignorado
    idade_grp = pd.cut(idade_v2, bins=[0, 19, 34, 100], labels=['Jovem (<20)', 'Adulta (20-34)', 'Madura (>34)'])

    res['raca_cor_mae'] = raca
    res['idade_mae'] = idade_grp.astype(object)
    res['idade_mae_anos'] = idade_v2
    res['escolaridade_mae'] = esc
    res['peso_g'] = peso_v2
    res['desfecho_nascimento'] = np.where(peso_v2 < 2500, 'Baixo Peso', 'Normal')
    res.loc[peso_v2.isna(), 'desfecho_nascimento'] = np.nan

    ausencia = {
        'RACACORMAE': raca.isna(), 'IDADEMAE': idade_v2.isna(),
        'ESCMAE': esc.isna(), 'PESO': peso_v2.isna(),
    }

    for raw_col, (new_col, labels) in CATEGORICAS.items():
        if raw_col in raw.columns:
            mapped = _clean(raw[raw_col]).map(labels)
        else:
            mapped = pd.Series(np.nan, index=raw.index)
        ausencia[raw_col] = mapped.isna()
        res[new_col] = mapped.fillna('Ignorado')

    for raw_col, (new_col, ignored_code) in NUMERICAS.items():
        if raw_col in raw.columns:
            num = pd.to_numeric(_clean(raw[raw_col]), errors='coerce')
            num = num.where(num != ignored_code)
        else:
            num = pd.Series(np.nan, index=raw.index)
        ausencia[raw_col] = num.isna()
        res[new_col] = num.astype('float32')

    masks = pd.DataFrame({
        'keep_v1': keep_v1,
        'baixo_peso_v1': desfecho_v1 == 'Baixo Peso',
        'sem_raca_cor': raca.isna(),
        'sem_idade': idade_v2.isna(),
        'idade_99_ou_invalida': idade.notna() & idade_v2.isna(),
        'sem_escolaridade': esc.isna(),
        'sem_peso': peso_v2.isna(),
    })
    masks['keep_v2'] = ~(masks['sem_raca_cor'] | masks['sem_idade'] | masks['sem_escolaridade'] | masks['sem_peso'])

    aus = pd.DataFrame(ausencia)
    aus['raca_cor_mae'] = raca.fillna('Sem raça/cor')
    aus_grp = aus.groupby('raca_cor_mae').agg(['sum', 'size'])
    return res, masks, aus_grp


def _fixed_categories():
    """Categorias fixas por coluna, para que bases de anos/UFs diferentes se concatenem como category."""
    idade_labels = ['Jovem (<20)', 'Adulta (20-34)', 'Madura (>34)']
    cats = {
        'uf': sorted(UF_IBGE.values()),
        'raca_cor_mae': list(RACA_MAP.values()),
        'idade_mae': idade_labels,
        'escolaridade_mae': list(ESC_MAP.values()),
        'desfecho_nascimento': ['Baixo Peso', 'Normal'],
    }
    for new_col, labels in CATEGORICAS.values():
        cats[new_col] = list(dict.fromkeys(labels.values())) + ['Ignorado']
    return cats


def _iter_sinasc_raw(year, source):
    """Gera (uf, DataFrame bruto) por UF de residência.

    source='dbc': arquivos DN{UF}{ano}.dbc do FTP do DATASUS em data/raw/sinasc/{ano}/ (usados na qualificação).
    source='csv': arquivo nacional do Portal de Dados Abertos do SUS em data/raw/opendatasus/sinasc/
                  (scripts/download_opendatasus.py); a UF vem dos 2 primeiros dígitos de CODMUNRES.
    """
    if source == 'dbc':
        raw_dir = os.path.join(PROJECT_ROOT, "data", "raw", "sinasc", str(year))
        files = sorted(glob.glob(os.path.join(raw_dir, f"DN??{year}.dbc")))
        if len(files) != len(UFS):
            print(f"[SINASC] AVISO: esperados {len(UFS)} arquivos .dbc em {raw_dir}, encontrados {len(files)}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            for path in files:
                yield os.path.basename(path)[2:4], _read_dbc(path, tmp_dir)
        return

    import zipfile
    sdir = os.path.join(OPENDATASUS_DIR, "sinasc")
    candidates = [os.path.join(sdir, f"SINASC_{year}_csv.zip"),
                  os.path.join(sdir, f"SINASC_{year}_preliminar_csv.zip")]
    zpath = next((c for c in candidates if os.path.exists(c)), None)
    if zpath is None:
        raise FileNotFoundError(f"SINASC {year}: nenhum de {candidates}. "
                                "Rode scripts/download_opendatasus.py --only sinasc")
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.lower().endswith('.csv'))
        raw = pd.read_csv(z.open(name), sep=';', dtype=str, encoding='latin-1', keep_default_na=False,
                          usecols=lambda c: c.upper() in SINASC_RAW_COLS)
    raw.columns = [c.upper() for c in raw.columns]
    uf = raw['CODMUNRES'].str.strip().str[:2].map(UF_IBGE)
    n_sem_uf = int(uf.isna().sum())
    if n_sem_uf:
        print(f"[SINASC] {year}: {n_sem_uf:,} registros com município de residência fora do Brasil ou "
              f"inválido -> fora da base (como nos arquivos por UF do FTP)")
    valid = uf.notna()
    for sigla, part in raw[valid].groupby(uf[valid], sort=True):
        yield sigla, part.reset_index(drop=True)


def build_sinasc(years=(2022,), source='csv'):
    """Gera um conjunto de arquivos por ano em data/processed/thesis/sinasc_v2/:
    sinasc_v2_{ano}.parquet, sinasc_v2_{ano}_exclusoes.csv e sinasc_v2_{ano}_ausencia.csv.

    Cada ano é processado, gravado e liberado da memória antes do seguinte.
    """
    out_dir = os.path.join(OUT_DIR, "sinasc_v2")
    os.makedirs(out_dir, exist_ok=True)
    cats = _fixed_categories()

    for year in years:
        print(f"\n[SINASC] ===== {year} (fonte: {source}) =====", flush=True)
        frames, mask_frames, aus_total = [], [], None
        for uf, raw in _iter_sinasc_raw(year, source):
            res, masks, aus_grp = _process_sinasc_uf(raw, uf)
            kept = res[masks['keep_v2']].copy()
            for col, labels in cats.items():
                unknown = set(kept[col].dropna().unique()) - set(labels)
                if unknown:
                    raise ValueError(f"{year}/{uf}: valores fora das categorias previstas em {col}: {unknown}")
                kept[col] = pd.Categorical(kept[col], categories=labels)
            frames.append(kept)
            mask_frames.append(masks)
            aus_total = aus_grp if aus_total is None else aus_total.add(aus_grp, fill_value=0)
            print(f"  {uf}: {len(raw):>9,} registros brutos | mantidos {int(masks['keep_v2'].sum()):>9,}",
                  flush=True)

        masks = pd.concat(mask_frames, ignore_index=True)
        total = len(masks)
        n_v1 = int(masks['keep_v1'].sum())
        print(f"[SINASC] {year}: {total:,} registros brutos; v2 mantém {int(masks['keep_v2'].sum()):,}")

        # ---- Conferência com a qualificação (só 2022) ----
        if year == 2022:
            ref = pd.read_parquet(os.path.join(PROCESSED_DIR, "sinasc_processed.parquet"))
            bp_v1 = int((masks['keep_v1'] & masks['baixo_peso_v1']).sum())
            ref_bp = int((ref['desfecho_nascimento'] == 'Baixo Peso').sum())
            print(f"[SINASC] regras da qualificação reaplicadas: {n_v1:,} linhas, {bp_v1:,} baixo peso "
                  f"(arquivo da qualificação: {len(ref):,} e {ref_bp:,}) -> "
                  f"{'CONFERE' if (n_v1 == len(ref) and bp_v1 == ref_bp) else 'DIFERE'}")
            del ref

        # ---- Exclusões do ano ----
        excl = [{'motivo': 'registros_brutos', 'registros': total, 'pct_do_total': 100.0}]
        for reason in ['sem_raca_cor', 'sem_idade', 'idade_99_ou_invalida', 'sem_escolaridade', 'sem_peso']:
            n = int(masks[reason].sum())
            excl.append({'motivo': reason, 'registros': n, 'pct_do_total': round(100 * n / total, 3)})
        n_excl = int((~masks['keep_v2']).sum())
        excl += [
            {'motivo': 'excluidos_por_qualquer_motivo', 'registros': n_excl,
             'pct_do_total': round(100 * n_excl / total, 3)},
            {'motivo': 'mantidos_v2', 'registros': int(masks['keep_v2'].sum()),
             'pct_do_total': round(100 * masks['keep_v2'].mean(), 3)},
        ]
        if year == 2022:
            excl.append({'motivo': 'mantidos_qualificacao', 'registros': n_v1,
                         'pct_do_total': round(100 * n_v1 / total, 3)})
        excl = pd.DataFrame(excl)
        excl_path = os.path.join(out_dir, f"sinasc_v2_{year}_exclusoes.csv")
        excl.to_csv(excl_path, index=False, encoding="utf-8-sig")
        print(excl.to_string(index=False))

        # ---- Ausência do ano, por variável bruta e raça/cor ----
        aus = []
        for var in aus_total.columns.get_level_values(0).unique():
            for raca_label, row in aus_total[var].iterrows():
                aus.append({'variavel': var, 'raca_cor_mae': raca_label,
                            'registros': int(row['size']), 'ausentes': int(row['sum']),
                            'pct_ausente': round(100 * row['sum'] / row['size'], 2)})
            tot_n, tot_a = aus_total[var]['size'].sum(), aus_total[var]['sum'].sum()
            aus.append({'variavel': var, 'raca_cor_mae': 'TODAS', 'registros': int(tot_n),
                        'ausentes': int(tot_a), 'pct_ausente': round(100 * tot_a / tot_n, 2)})
        aus_path = os.path.join(out_dir, f"sinasc_v2_{year}_ausencia.csv")
        pd.DataFrame(aus).to_csv(aus_path, index=False, encoding="utf-8-sig")

        # ---- Base do ano ----
        out = pd.concat(frames, ignore_index=True)
        del frames, mask_frames, masks
        out['idade_mae_anos'] = out['idade_mae_anos'].astype('int16')
        out['peso_g'] = out['peso_g'].astype('int16')
        path = os.path.join(out_dir, f"sinasc_v2_{year}.parquet")
        out.to_parquet(path, index=False, compression="zstd")
        print(f"[SINASC] {year}: {len(out):,} linhas x {out.shape[1]} colunas, baixo peso "
              f"{100 * (out['desfecho_nascimento'] == 'Baixo Peso').mean():.2f}% -> {path} "
              f"({os.path.getsize(path) / 1e6:.1f} MB)")
        print(f"[SINASC] relatórios: {excl_path} e {aus_path}")
        del out


# ---------------------------------------------------------------------------
# SIH — internações de um ano (AIH reduzida, FTP do DATASUS)
# ---------------------------------------------------------------------------

DATASUS_FTP_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "datasus_ftp")

# Capítulos da CID-10 (faixas de categorias de 3 caracteres, inclusivas).
CID10_CAPITULOS = [
    (1, 'A00', 'B99', 'I – Infecciosas e parasitárias (A00–B99)'),
    (2, 'C00', 'D48', 'II – Neoplasias (C00–D48)'),
    (3, 'D50', 'D89', 'III – Sangue e transtornos imunitários (D50–D89)'),
    (4, 'E00', 'E90', 'IV – Endócrinas, nutricionais e metabólicas (E00–E90)'),
    (5, 'F00', 'F99', 'V – Transtornos mentais e comportamentais (F00–F99)'),
    (6, 'G00', 'G99', 'VI – Sistema nervoso (G00–G99)'),
    (7, 'H00', 'H59', 'VII – Olho e anexos (H00–H59)'),
    (8, 'H60', 'H95', 'VIII – Ouvido e apófise mastoide (H60–H95)'),
    (9, 'I00', 'I99', 'IX – Aparelho circulatório (I00–I99)'),
    (10, 'J00', 'J99', 'X – Aparelho respiratório (J00–J99)'),
    (11, 'K00', 'K93', 'XI – Aparelho digestivo (K00–K93)'),
    (12, 'L00', 'L99', 'XII – Pele e tecido subcutâneo (L00–L99)'),
    (13, 'M00', 'M99', 'XIII – Osteomuscular e tecido conjuntivo (M00–M99)'),
    (14, 'N00', 'N99', 'XIV – Aparelho geniturinário (N00–N99)'),
    (15, 'O00', 'O99', 'XV – Gravidez, parto e puerpério (O00–O99)'),
    (16, 'P00', 'P96', 'XVI – Afecções do período perinatal (P00–P96)'),
    (17, 'Q00', 'Q99', 'XVII – Malformações congênitas e anomalias cromossômicas (Q00–Q99)'),
    (18, 'R00', 'R99', 'XVIII – Sintomas, sinais e achados anormais (R00–R99)'),
    (19, 'S00', 'T98', 'XIX – Lesões, envenenamentos e causas externas (S00–T98)'),
    (20, 'V01', 'Y98', 'XX – Causas externas de morbidade e mortalidade (V01–Y98)'),
    (21, 'Z00', 'Z99', 'XXI – Fatores que influenciam o contato com serviços de saúde (Z00–Z99)'),
    (22, 'U00', 'U99', 'XXII – Códigos para propósitos especiais (U00–U99)'),
]
CID10_CAPITULO_LABEL = {n: label for n, _, _, label in CID10_CAPITULOS}

# Rótulos dos códigos do SIH (layout da AIH reduzida, RD).
SIH_SEXO = {'1': 'Masculino', '3': 'Feminino'}
SIH_RACA = {'01': 'Branca', '02': 'Preta', '03': 'Parda', '04': 'Amarela', '05': 'Indígena'}  # 99 = sem informação
SIH_CARATER = {'01': 'Eletiva', '02': 'Urgência', '03': 'Acidente de trabalho', '04': 'Acidente de trabalho',
               '05': 'Acidente de trânsito', '06': 'Outras lesões e envenenamentos'}
SIH_COMPLEX = {'02': 'Média complexidade', '03': 'Alta complexidade'}
SIH_ESPEC = {'01': 'Cirúrgico', '02': 'Obstétrico', '03': 'Clínico', '04': 'Crônico', '05': 'Psiquiatria',
             '06': 'Pneumologia sanitária', '07': 'Pediátrico', '08': 'Reabilitação'}  # demais: hospital-dia/outros
SIH_FAIXA = ['Criança/adolescente (0-17)', 'Adulto (18-59)', 'Idoso (60+)']
# Motivo de saída (COBRANCA), pelo primeiro dígito: 1 alta; 2 permanência; 3 transferência; 4 óbito;
# 5 encerramento administrativo; 6 alta ou óbito da mãe/puérpera (obstetrícia). O óbito vem de MORTE.
SIH_SAIDA_EXCLUIDA = {'2': 'saida_permanencia', '3': 'saida_transferencia', '5': 'saida_encerramento_administrativo'}
# Capítulos que entram no resumo (para documentar o funil) mas não ganham arquivo nem entram no painel:
# óbito raro demais (desfecho sem variação suficiente para a análise de viés) ou poucas internações
# (grupos interseccionais pequenos demais). Em 2024: XV, VII e VIII pela taxa; XXII pelo tamanho.
SIH_OBITO_MIN_PCT = 0.2
SIH_N_MIN = 1000
SIH_RAW_COLS = ['N_AIH', 'IDENT', 'DT_INTER', 'SEXO', 'RACA_COR', 'IDADE', 'COD_IDADE', 'DIAG_PRINC',
                'CAR_INT', 'COMPLEX', 'ESPEC', 'MUNIC_RES', 'MUNIC_MOV', 'COBRANCA', 'MORTE']


def _read_dbf_columns(dbf_path, cols):
    """Lê só as colunas pedidas de um DBF (registros de largura fixa), como texto sem espaços.

    Equivalente ao dbfread para campos de texto/número, mas fatiando o arquivo inteiro com numpy:
    dezenas de vezes mais rápido para os 113 campos da AIH, dos quais usamos 15.
    """
    with open(dbf_path, 'rb') as f:
        data = f.read()
    n = int.from_bytes(data[4:8], 'little')
    header_len = int.from_bytes(data[8:10], 'little')
    rec_len = int.from_bytes(data[10:12], 'little')
    fields, offset, pos = {}, 1, 32  # byte 0 de cada registro = marca de exclusão
    while data[pos] != 0x0D:
        name = data[pos:pos + 11].split(b'\0')[0].decode('latin-1').upper()
        length = data[pos + 16]
        fields[name] = (offset, length)
        offset += length
        pos += 32
    rec = np.frombuffer(data, dtype=np.uint8, count=n * rec_len, offset=header_len).reshape(n, rec_len)
    out = {}
    for col in cols:
        o, length = fields[col]
        raw = np.ascontiguousarray(rec[:, o:o + length]).view(f'S{length}').ravel()
        out[col] = pd.Series(raw).str.decode('latin-1').str.strip()
    df = pd.DataFrame(out)
    return df[rec[:, 0] != ord('*')].reset_index(drop=True)


def _sih_categories():
    return {
        'uf': sorted(UF_IBGE.values()),
        'sexo': ['Masculino', 'Feminino'],
        'raca_cor': ['Branca', 'Preta', 'Parda', 'Amarela', 'Indígena'],
        'faixa_etaria': SIH_FAIXA,
        'desfecho': ['Óbito', 'Alta'],
        'capitulo_cid': [label for _, _, _, label in CID10_CAPITULOS],
        'carater_internacao': list(dict.fromkeys(SIH_CARATER.values())) + ['Ignorado'],
        'complexidade': list(SIH_COMPLEX.values()) + ['Ignorado'],
        'especialidade_leito': list(SIH_ESPEC.values()) + ['Hospital-dia/outros'],
    }


def _sih_compact(raw, uf, comp, year):
    """AIHs com internação no ano `year`, em tipos compactos (o ano inteiro tem ~12 milhões de linhas)."""
    raw = raw[raw['DT_INTER'].str[:4] == str(year)]
    d = pd.DataFrame(index=raw.index)
    d['n_aih'] = pd.to_numeric(raw['N_AIH'], errors='coerce').astype('Int64')
    d['competencia'] = np.int16(comp)
    d['longa_permanencia'] = raw['IDENT'].eq('5')
    d['mes_internacao'] = pd.to_numeric(raw['DT_INTER'].str[4:6], errors='coerce').astype('Int8')
    d['uf'] = uf
    d['sexo'] = raw['SEXO'].map(SIH_SEXO)
    d['raca_cor'] = raw['RACA_COR'].map(SIH_RACA)
    idade = pd.to_numeric(raw['IDADE'], errors='coerce')
    cod = raw['COD_IDADE']
    # COD_IDADE: 2 = dias, 3 = meses, 4 = anos, 5 = centenas de anos (idade = 100 + IDADE).
    anos = np.select([cod.eq('4'), cod.eq('5'), cod.eq('3'), cod.isin(['0', '1', '2'])],
                     [idade, 100 + idade, idade // 12, 0], default=np.nan)
    d['idade_anos'] = pd.Series(anos, index=raw.index).astype('Int16')
    d['cid_categoria'] = raw['DIAG_PRINC'].str[:3]
    cap = pd.Series(pd.NA, index=raw.index, dtype='Int8')
    for n, ini, fim, _ in CID10_CAPITULOS:
        cap[(d['cid_categoria'] >= ini) & (d['cid_categoria'] <= fim)] = n
    d['capitulo_n'] = cap
    d['carater_internacao'] = raw['CAR_INT'].map(SIH_CARATER).fillna('Ignorado')
    d['complexidade'] = raw['COMPLEX'].map(SIH_COMPLEX).fillna('Ignorado')
    d['especialidade_leito'] = raw['ESPEC'].map(SIH_ESPEC).fillna('Hospital-dia/outros')
    d['fora_municipio_residencia'] = raw['MUNIC_RES'].str[:6] != raw['MUNIC_MOV'].str[:6]
    d['saida_grupo'] = raw['COBRANCA'].str[:1]
    d['obito'] = raw['MORTE'].eq('1')
    for col in ['uf', 'sexo', 'raca_cor', 'cid_categoria', 'carater_internacao', 'complexidade',
                'especialidade_leito', 'saida_grupo']:
        d[col] = d[col].astype('category')
    return d.reset_index(drop=True)


def build_sih(year=2024, last_comp='2025-06'):
    """Internações com data de internação em `year`, de todas as competências até `last_comp`.

    Grava em data/processed/thesis/sih_v2/{ano}/ um parquet por capítulo da CID-10
    (sih_v2_{ano}_cap{NN}.parquet), o funil de exclusões, a cobertura por competência e um
    resumo por capítulo (sih_v2_{ano}_resumo.parquet, lido pelo painel).
    """
    sdir = os.path.join(DATASUS_FTP_DIR, "sih")
    y_end, m_end = map(int, last_comp.split('-'))
    comps, y, m = [], year, 1
    while (y, m) <= (y_end, m_end):
        comps.append(f"{y % 100:02d}{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)

    frames, cobertura, faltando = [], [], []
    with tempfile.TemporaryDirectory() as tmp_dir:
        import pyreaddbc
        for comp in comps:
            n_comp = 0
            for uf in UFS:
                path = os.path.join(sdir, f"RD{uf}{comp}.dbc")
                if not os.path.exists(path):
                    faltando.append(os.path.basename(path))
                    continue
                dbf = os.path.join(tmp_dir, f"RD{uf}{comp}.dbf")
                pyreaddbc.dbc2dbf(path, dbf)
                part = _sih_compact(_read_dbf_columns(dbf, SIH_RAW_COLS), uf, int(comp), year)
                os.remove(dbf)
                n_comp += len(part)
                frames.append(part)
            cobertura.append({'competencia': f"20{comp[:2]}-{comp[2:]}", 'aihs_internacao_no_ano': n_comp})
            print(f"[SIH] competência 20{comp[:2]}-{comp[2:]}: {n_comp:,} AIHs com internação em {year}",
                  flush=True)
    if faltando:
        raise FileNotFoundError(f"[SIH] {len(faltando)} arquivos ausentes (ex.: {faltando[:3]}). "
                                "Rode scripts/download_datasus_ftp.py")

    cats = _sih_categories()
    for col in ['uf', 'sexo', 'raca_cor', 'carater_internacao', 'complexidade', 'especialidade_leito']:
        for f in frames:
            f[col] = pd.Categorical(f[col].astype(object), categories=cats[col])
    all_cid = sorted(set().union(*[set(f['cid_categoria'].cat.categories) for f in frames]))
    all_saida = sorted(set().union(*[set(f['saida_grupo'].cat.categories) for f in frames]))
    for f in frames:
        f['cid_categoria'] = pd.Categorical(f['cid_categoria'].astype(object), categories=all_cid)
        f['saida_grupo'] = pd.Categorical(f['saida_grupo'].astype(object), categories=all_saida)
    d = pd.concat(frames, ignore_index=True)
    del frames

    # ---- Funil de exclusões (sequencial) ----
    funil = []

    def step(motivo, mask):
        nonlocal d
        n = int(mask.sum())
        by_cap = d.loc[mask, 'capitulo_n'].value_counts(dropna=False)
        funil.append({'motivo': motivo, 'registros': n, **{f'cap{int(k):02d}' if pd.notna(k) else 'cap_invalido': int(v)
                                                        for k, v in by_cap.items()}})
        d = d[~mask].reset_index(drop=True)

    total = len(d)
    funil.append({'motivo': 'aihs_com_internacao_no_ano', 'registros': total,
                  **{f'cap{int(k):02d}' if pd.notna(k) else 'cap_invalido': int(v)
                     for k, v in d['capitulo_n'].value_counts(dropna=False).items()}})
    # A mesma AIH pode ser reapresentada em outra competência: fica a mais recente.
    d = d.sort_values('competencia', kind='stable')
    step('aih_repetida_em_outra_competencia', d.duplicated('n_aih', keep='last').to_numpy())
    step('aih_de_longa_permanencia (continuação)', d['longa_permanencia'].to_numpy())
    for code, motivo in SIH_SAIDA_EXCLUIDA.items():
        step(motivo, d['saida_grupo'].astype(object).eq(code).to_numpy())
    step('sem_raca_cor', d['raca_cor'].isna().to_numpy())
    step('sexo_invalido', d['sexo'].isna().to_numpy())
    step('idade_invalida', (d['idade_anos'].isna() | (d['idade_anos'] > 120)).to_numpy())
    step('cid_fora_dos_capitulos', d['capitulo_n'].isna().to_numpy())
    funil.append({'motivo': 'mantidos', 'registros': len(d),
                  **{f'cap{int(k):02d}': int(v) for k, v in d['capitulo_n'].value_counts().items()}})
    funil = pd.DataFrame(funil).fillna(0)
    funil.insert(2, 'pct_do_total', (100 * funil['registros'] / total).round(3))

    # ---- Base final ----
    d['faixa_etaria'] = pd.Categorical(
        pd.cut(d['idade_anos'].astype(float), bins=[-1, 17, 59, 200], labels=SIH_FAIXA).astype(object),
        categories=SIH_FAIXA)
    d['desfecho'] = pd.Categorical(np.where(d['obito'], 'Óbito', 'Alta'), categories=cats['desfecho'])
    d['capitulo_cid'] = pd.Categorical(d['capitulo_n'].map(CID10_CAPITULO_LABEL), categories=cats['capitulo_cid'])
    d['idade_anos'] = d['idade_anos'].astype('int16')
    d['mes_internacao'] = d['mes_internacao'].astype('int8')
    cols = ['uf', 'sexo', 'raca_cor', 'faixa_etaria', 'idade_anos', 'desfecho', 'capitulo_cid', 'cid_categoria',
            'carater_internacao', 'complexidade', 'especialidade_leito', 'fora_municipio_residencia',
            'mes_internacao']

    out_dir = os.path.join(OUT_DIR, "sih_v2", str(year))
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):  # arquivos de capítulo de uma geração anterior
        if f.startswith(f"sih_v2_{year}_cap") and f.endswith(".parquet"):
            os.remove(os.path.join(out_dir, f))
    resumo = []
    for n, _, _, label in CID10_CAPITULOS:
        part = d.loc[d['capitulo_n'] == n, cols].reset_index(drop=True)
        brutos = int(funil.loc[funil['motivo'] == 'aihs_com_internacao_no_ano', f'cap{n:02d}'].sum()) \
            if f'cap{n:02d}' in funil else 0
        obito_pct = 100 * (part['desfecho'] == 'Óbito').mean() if len(part) else np.nan
        if len(part) == 0:
            motivo_fora = 'sem internações'
        elif len(part) < SIH_N_MIN:
            motivo_fora = f'menos de {SIH_N_MIN} internações'
        elif obito_pct < SIH_OBITO_MIN_PCT:
            motivo_fora = f'óbito abaixo de {SIH_OBITO_MIN_PCT}%'
        else:
            motivo_fora = ''
        incluido = motivo_fora == ''
        resumo.append({'capitulo_n': n, 'capitulo_cid': label, 'aihs_no_ano': brutos, 'mantidos': len(part),
                       'obito_pct': round(obito_pct, 3), 'incluido': incluido, 'motivo_fora': motivo_fora})
        if not incluido:
            continue
        part['cid_categoria'] = part['cid_categoria'].cat.remove_unused_categories()
        part.to_parquet(os.path.join(out_dir, f"sih_v2_{year}_cap{n:02d}.parquet"), index=False, compression="zstd")
    resumo = pd.DataFrame(resumo)
    resumo.to_parquet(os.path.join(out_dir, f"sih_v2_{year}_resumo.parquet"), index=False)
    funil.to_csv(os.path.join(out_dir, f"sih_v2_{year}_exclusoes.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(cobertura).to_csv(os.path.join(out_dir, f"sih_v2_{year}_cobertura_competencias.csv"),
                                   index=False, encoding="utf-8-sig")

    print("\n[SIH] funil:")
    print(funil[['motivo', 'registros', 'pct_do_total']].to_string(index=False))
    print(f"\n[SIH] {len(d):,} internações mantidas; óbito {100 * d['obito'].mean():.2f}%")
    print(resumo.to_string(index=False))
    sizes = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
    print(f"[SIH] arquivos em {out_dir} ({sizes / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Build the thesis versions of the datasets.")
    parser.add_argument("--only", nargs="+", choices=["compas", "synthetic", "sinasc", "sih"],
                        default=["compas", "synthetic", "sinasc"])
    parser.add_argument("--sinasc-years", nargs="+", type=int, default=[2022, 2023, 2024],
                        help="Anos do SINASC (padrão: 2022 2023 2024, os consolidados).")
    parser.add_argument("--sinasc-source", choices=["csv", "dbc"], default="csv",
                        help="csv = Portal de Dados Abertos do SUS (padrão); dbc = FTP do DATASUS.")
    parser.add_argument("--sih-year", type=int, default=2024,
                        help="Ano de internação do SIH (padrão: 2024).")
    parser.add_argument("--sih-last-comp", default="2025-06",
                        help="Última competência lida, AAAA-MM (AIHs de internações do ano chegam depois dele).")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if "compas" in args.only:
        build_compas()
    if "synthetic" in args.only:
        build_synthetic()
    if "sinasc" in args.only:
        build_sinasc(years=args.sinasc_years, source=args.sinasc_source)
    if "sih" in args.only:
        build_sih(year=args.sih_year, last_comp=args.sih_last_comp)


if __name__ == "__main__":
    main()
