"""
collect_results.py — Junta os shards de jobs independentes
===========================================================

Lê os shards gravados por `run_experiments.py --shard-dir ...` e produz, em
--out-dir, arquivos parquet com o mesmo nome e esquema da execução sequencial
(mais uma coluna `seed`), prontos para o dashboard.

Depende apenas de pandas/pyarrow: pode rodar no nó de login do cluster ou
localmente, depois de copiar a pasta de shards.

Uso
---
  python scripts/collect_results.py --shard-dir runs/exp1/shards --out-dir runs/exp1/results
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.experiment_io import collect_shards

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser(description="Merge experiment shards into consolidated results.")
    parser.add_argument("--shard-dir", required=True, help="Directory with *.metrics.parquet shards.")
    parser.add_argument("--out-dir", required=True, help="Where consolidated parquet files are written.")
    parser.add_argument("--expected-folds", type=int, default=3,
                        help="Outer folds expected per configuration (default: 3). 0 disables the check.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with an error if any configuration is missing folds.")
    args = parser.parse_args()

    default_results = os.path.join(PROJECT_ROOT, "data", "results")
    if os.path.abspath(args.out_dir) == os.path.abspath(default_results):
        parser.error("--out-dir aponta para data/results, que guarda os resultados da qualificação. "
                     "Escolha outra pasta.")

    summary = collect_shards(args.shard_dir, args.out_dir, expected_folds=args.expected_folds or None)

    print(f"\n[DONE] {summary['shards']} shards -> {args.out_dir}")
    print(f"       Rows in agg results      : {summary['agg_rows']}")
    print(f"       Rows in subgroup results : {summary['subgroup_rows']}")
    if summary["incomplete"]:
        print(f"\n[WARNING] {len(summary['incomplete'])} configuração(ões) incompleta(s):")
        for line in summary["incomplete"][:50]:
            print(f"  - {line}")
        if args.strict:
            sys.exit(1)


if __name__ == "__main__":
    main()
