"""
download_datasus_ftp.py — Baixa arquivos mensais do SIH (AIH reduzida) do FTP do DATASUS
=========================================================================================

Fonte: ftp://ftp.datasus.gov.br/dissemin/publicos/SIHSUS/200801_/Dados/RD{UF}{AAMM}.dbc
(AAMM = mês de competência, isto é, o mês em que a AIH foi processada, não o da internação).

Para um ano de internações completo, é preciso baixar também os meses seguintes: AIHs de
internações de dezembro, por exemplo, só são processadas em janeiro ou depois.

Cada arquivo baixado é registrado em data/raw/datasus_ftp/MANIFEST.csv com caminho no FTP,
data de extração, data de modificação no servidor, tamanho e SHA-256 (proveniência).

Uso
---
  python scripts/download_datasus_ftp.py --inicio 2024-01 --fim 2025-06
  python scripts/download_datasus_ftp.py --inicio 2024-06 --fim 2024-06 --ufs RS SC

Arquivos já presentes com o mesmo tamanho não são baixados de novo.
"""

import argparse
import csv
import datetime as dt
import hashlib
import os
import sys
import time
from ftplib import FTP, error_perm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "datasus_ftp")
MANIFEST = os.path.join(RAW_DIR, "MANIFEST.csv")
HOST = "ftp.datasus.gov.br"
SIH_DIR = "/dissemin/publicos/SIHSUS/200801_/Dados"
UFS = ['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR',
       'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO']
MANIFEST_COLS = ["sistema", "arquivo", "caminho_ftp", "extraido_em", "modificado_no_servidor", "bytes", "sha256"]

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


def _months(inicio, fim):
    y, m = map(int, inicio.split("-"))
    y_end, m_end = map(int, fim.split("-"))
    while (y, m) <= (y_end, m_end):
        yield f"{y % 100:02d}{m:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def _connect():
    ftp = FTP(HOST, timeout=120)
    ftp.login()
    ftp.cwd(SIH_DIR)
    return ftp


def download_sih(ftp, nome, manifest):
    dest_dir = os.path.join(RAW_DIR, "sih")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, nome)
    size = ftp.size(nome)
    try:
        modified = ftp.sendcmd(f"MDTM {nome}").split()[-1]
    except error_perm:
        modified = ""
    if os.path.exists(dest) and os.path.getsize(dest) == size and ("sih", nome) in manifest:
        return "ok"
    tmp = dest + ".part"
    with open(tmp, "wb") as out:
        ftp.retrbinary(f"RETR {nome}", out.write)
    got = os.path.getsize(tmp)
    if got != size:
        os.remove(tmp)
        raise IOError(f"{nome}: tamanho {got} difere do anunciado {size}")
    os.replace(tmp, dest)
    manifest[("sih", nome)] = {
        "sistema": "sih", "arquivo": nome, "caminho_ftp": f"{HOST}{SIH_DIR}/{nome}",
        "extraido_em": dt.datetime.now().isoformat(timespec="seconds"),
        "modificado_no_servidor": modified, "bytes": got, "sha256": _sha256(dest),
    }
    _write_manifest(manifest)
    return "baixado"


def main():
    parser = argparse.ArgumentParser(description="Download monthly SIH (RD) files from the DATASUS FTP.")
    parser.add_argument("--inicio", required=True, help="Primeiro mês de competência, AAAA-MM.")
    parser.add_argument("--fim", required=True, help="Último mês de competência, AAAA-MM.")
    parser.add_argument("--ufs", nargs="+", default=UFS, choices=UFS)
    args = parser.parse_args()

    manifest = _read_manifest()
    ftp = _connect()
    falhas = []
    for aamm in _months(args.inicio, args.fim):
        total = 0
        for uf in args.ufs:
            nome = f"RD{uf}{aamm}.dbc"
            for tentativa in range(3):  # o FTP do DATASUS derruba conexões longas de vez em quando
                try:
                    download_sih(ftp, nome, manifest)
                    total += os.path.getsize(os.path.join(RAW_DIR, "sih", nome))
                    break
                except error_perm as e:  # arquivo inexistente: não adianta tentar de novo
                    print(f"  [ERRO] {nome}: {e}")
                    falhas.append(nome)
                    break
                except Exception as e:
                    print(f"  [..] {nome}: {e}; reconectando ({tentativa + 1}/3)")
                    time.sleep(5)
                    try:
                        ftp.close()
                    except Exception:
                        pass
                    ftp = _connect()
            else:
                falhas.append(nome)
        print(f"competência {aamm}: {total / 1e6:.1f} MB", flush=True)
    ftp.quit()
    print(f"\nManifesto: {MANIFEST}")
    if falhas:
        print("Falharam:", ", ".join(falhas))
        sys.exit(1)


if __name__ == "__main__":
    main()
