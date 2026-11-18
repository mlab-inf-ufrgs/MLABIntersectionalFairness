"""
download_opendatasus.py — Baixa microdados do Portal de Dados Abertos do SUS
=============================================================================

Fonte: https://dadosabertos.saude.gov.br/dataset (arquivos servidos pelo bucket
S3 público ckan.saude.gov.br e, no Vigitel, por svs.aids.gov.br).

Cada arquivo baixado é registrado em data/raw/opendatasus/MANIFEST.csv com URL,
data de extração, data de modificação no servidor, tamanho e SHA-256 — o passo 1
do protocolo de rigor (proveniência).

Uso
---
  python scripts/download_opendatasus.py                      # tudo o que está em CATALOGO
  python scripts/download_opendatasus.py --only sinasc sim    # só alguns sistemas
  python scripts/download_opendatasus.py --list               # mostra o catálogo sem baixar

Arquivos já presentes com o mesmo tamanho não são baixados de novo.
"""

import argparse
import csv
import datetime as dt
import hashlib
import os
import sys
import urllib.request

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "opendatasus")
MANIFEST = os.path.join(RAW_DIR, "MANIFEST.csv")

S3 = "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br"

# sistema -> lista de (nome local, URL). "preliminar" no nome marca anos ainda não consolidados.
CATALOGO = {
    "sinasc": [
        ("dicionario_sinasc.pdf", "https://diaad.s3.sa-east-1.amazonaws.com/sinasc/SINASC+-+Estrutura.pdf"),
        ("SINASC_2022_csv.zip", f"{S3}/SINASC/csv/SINASC_2022_csv.zip"),
        ("SINASC_2023_csv.zip", f"{S3}/SINASC/csv/SINASC_2023_csv.zip"),
        ("SINASC_2024_csv.zip", f"{S3}/SINASC/csv/SINASC_2024_csv.zip"),
        ("SINASC_2025_preliminar_csv.zip", f"{S3}/SINASC/csv/SINASC_2025_csv.zip"),
    ],
    "sim": [
        ("dicionario_sim_2025.pdf", f"{S3}/SIM/Dicionario_SIM_2025.pdf"),
        ("Mortalidade_Geral_2022_csv.zip", f"{S3}/SIM/csv/Mortalidade_Geral_2022_csv.zip"),
        ("Mortalidade_Geral_2023_csv.zip", f"{S3}/SIM/csv/Mortalidade_Geral_2023_csv.zip"),
        ("Mortalidade_Geral_2024_csv.zip", f"{S3}/SIM/csv/Mortalidade_Geral_2024_csv.zip"),
    ],
    "dengue": [
        ("dicionario_dengue.pdf", f"{S3}/SINAN/Dengue/dic_dados_dengue.pdf"),
        ("DENGBR22.csv.zip", f"{S3}/SINAN/Dengue/csv/DENGBR22.csv.zip"),
        ("DENGBR23.csv.zip", f"{S3}/SINAN/Dengue/csv/DENGBR23.csv.zip"),
        ("DENGBR24.csv.zip", f"{S3}/SINAN/Dengue/csv/DENGBR24.csv.zip"),
        ("DENGBR25.csv.zip", f"{S3}/SINAN/Dengue/csv/DENGBR25.csv.zip"),
    ],
    "srag": [
        ("dicionario_srag_2019_2025.pdf", f"{S3}/SRAG/dicionario-de-dados-2019-a-2025.pdf"),
        ("INFLUD23-23-03-2026.parquet", f"{S3}/SRAG/2023/INFLUD23-23-03-2026.parquet"),
        ("INFLUD24-23-03-2026.parquet", f"{S3}/SRAG/2024/INFLUD24-23-03-2026.parquet"),
        ("INFLUD25-16-11-2026.parquet", f"{S3}/SRAG/2025/INFLUD25-16-11-2026.parquet"),
    ],
    "vigitel": [
        ("dicionario-vigitel-2006-2024.xlsx", "https://svs.aids.gov.br/daent/cgdnt/vigitel/dicionario-vigitel-2006-2024.xlsx"),
        ("vigitel-2006-2024-peso-rake-csv.zip", "https://svs.aids.gov.br/daent/cgdnt/vigitel/vigitel-2006-2024-peso-rake-csv.zip"),
    ],
}

MANIFEST_COLS = ["sistema", "arquivo", "url", "extraido_em", "modificado_no_servidor", "bytes", "sha256"]

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _read_manifest():
    if not os.path.exists(MANIFEST):
        return {}
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return {(r["sistema"], r["arquivo"]): r for r in csv.DictReader(f)}


def _write_manifest(rows):
    os.makedirs(RAW_DIR, exist_ok=True)
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        for key in sorted(rows):
            w.writerow(rows[key])
    os.replace(tmp, MANIFEST)


def download(sistema, nome, url, manifest):
    dest_dir = os.path.join(RAW_DIR, sistema)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, nome)

    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "research-data-client"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        size = int(resp.headers.get("Content-Length") or -1)
        modified = resp.headers.get("Last-Modified", "")

    if os.path.exists(dest) and os.path.getsize(dest) == size and (sistema, nome) in manifest:
        print(f"  [ok] {sistema}/{nome} já baixado")
        return

    print(f"  [..] {sistema}/{nome} ({size / 1e6:.1f} MB)")
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "research-data-client"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as out:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
    got = os.path.getsize(tmp)
    if size >= 0 and got != size:
        os.remove(tmp)
        raise IOError(f"{nome}: tamanho {got} difere do anunciado {size}")
    os.replace(tmp, dest)

    manifest[(sistema, nome)] = {
        "sistema": sistema, "arquivo": nome, "url": url,
        "extraido_em": dt.datetime.now().isoformat(timespec="seconds"),
        "modificado_no_servidor": modified, "bytes": got, "sha256": _sha256(dest),
    }
    _write_manifest(manifest)


def main():
    parser = argparse.ArgumentParser(description="Download SUS open microdata with a provenance manifest.")
    parser.add_argument("--only", nargs="+", choices=sorted(CATALOGO), help="Systems to download.")
    parser.add_argument("--list", action="store_true", help="Print the catalogue and exit.")
    args = parser.parse_args()

    sistemas = args.only or list(CATALOGO)
    if args.list:
        for s in sistemas:
            for nome, url in CATALOGO[s]:
                print(f"{s:8s} {nome:40s} {url}")
        return

    manifest = _read_manifest()
    falhas = []
    for s in sistemas:
        print(f"\n== {s}")
        for nome, url in CATALOGO[s]:
            try:
                download(s, nome, url, manifest)
            except Exception as e:  # um arquivo indisponível não interrompe os demais
                print(f"  [ERRO] {nome}: {e}")
                falhas.append(f"{s}/{nome}")
    print(f"\nManifesto: {MANIFEST}")
    if falhas:
        print("Falharam:", ", ".join(falhas))
        sys.exit(1)


if __name__ == "__main__":
    main()
