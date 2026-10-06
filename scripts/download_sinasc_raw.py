"""
Baixa os microdados brutos do SINASC (arquivos .dbc, um por UF) do FTP do DATASUS.

Diferente de fetch_datasus_ftp.py, este script NÃO converte nem apaga os .dbc:
apenas baixa os arquivos originais e gera um manifesto com tamanho e SHA-256
de cada um, para que possam ser repassados e conferidos por outras pessoas.

Uso:
    python scripts/download_sinasc_raw.py            # ano 2022
    python scripts/download_sinasc_raw.py --ano 2021

Requer apenas a biblioteca padrão do Python e acesso à porta 21 (FTP).
"""
import argparse
import hashlib
import os
import time
from ftplib import FTP

FTP_HOST = 'ftp.datasus.gov.br'
FTP_DIR = '/dissemin/publicos/SINASC/1996_/Dados/DNRES'

UFS = ['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA',
       'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN',
       'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(ftp, filename, out_path, tentativas=3):
    for i in range(1, tentativas + 1):
        try:
            remote_size = ftp.size(filename)
            if os.path.exists(out_path) and os.path.getsize(out_path) == remote_size:
                print(f"  {filename}: já existe, pulando")
                return
            with open(out_path + '.part', 'wb') as f:
                ftp.retrbinary(f'RETR {filename}', f.write)
            if os.path.getsize(out_path + '.part') != remote_size:
                raise IOError('tamanho baixado difere do servidor')
            os.replace(out_path + '.part', out_path)
            print(f"  {filename}: {remote_size / 1e6:.1f} MB")
            return
        except Exception as e:
            print(f"  {filename}: tentativa {i} falhou ({e})")
            time.sleep(5 * i)
    raise RuntimeError(f'Falha ao baixar {filename}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ano', type=int, default=2022)
    parser.add_argument('--saida', default=os.path.join('data', 'raw', 'sinasc'))
    args = parser.parse_args()

    out_dir = os.path.join(args.saida, str(args.ano))
    os.makedirs(out_dir, exist_ok=True)

    ftp = FTP(FTP_HOST, timeout=60)
    ftp.login()
    ftp.cwd(FTP_DIR)
    print(f"Conectado a {FTP_HOST}{FTP_DIR}")

    for uf in UFS:
        filename = f"DN{uf}{args.ano}.dbc"
        download(ftp, filename, os.path.join(out_dir, filename))
    ftp.quit()

    manifesto = os.path.join(out_dir, 'MANIFESTO.sha256')
    with open(manifesto, 'w', encoding='utf-8') as f:
        for uf in UFS:
            filename = f"DN{uf}{args.ano}.dbc"
            f.write(f"{sha256(os.path.join(out_dir, filename))}  {filename}\n")
    print(f"Concluído: {len(UFS)} arquivos em {out_dir} (manifesto em {manifesto})")


if __name__ == '__main__':
    main()
