#!/bin/bash
# Experimentos do SIH 2024 no PCAD: um job do array = uma unidade de trabalho
# (capítulo x modelo x configuração x dobra x semente). Unidades já concluídas são puladas,
# então o array pode ser reenviado sem refazer nada.
#
# Uso (na raiz do repositório, no PCAD):
#   python scripts/run_experiments.py --registry thesis --print-units \
#       --datasets $(python -c "from utils.thesis_datasets import THESIS_DATASETS as T; print(' '.join(k for k in T if k.startswith('sih_v2_2024')))") \
#       --seeds 42 > pcad/units_sih.txt
#   mkdir -p runs/sih2024/logs        # o Slurm não cria a pasta dos logs
#   N=$(wc -l < pcad/units_sih.txt)
#   sbatch --array=1-${N}%20 pcad/sih_array.slurm
#   # ao final:
#   python scripts/collect_results.py --shard-dir runs/sih2024/shards --out-dir runs/sih2024/results
#
# RASCUNHO: conferir com a documentação do PCAD o limite de tempo de cada partição,
# o tamanho máximo de array e como ativar o ambiente Python (módulo ou conda).

#SBATCH --job-name=sih2024
#SBATCH --partition=hype,tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=runs/sih2024/logs/%A_%a.out
#SBATCH --error=runs/sih2024/logs/%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# Ambiente Python (ajustar ao PCAD)
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate prop_tese

UNIT_ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" pcad/units_sih.txt)
echo "[unit ${SLURM_ARRAY_TASK_ID}] ${UNIT_ARGS} @ $(hostname)"

python scripts/run_experiments.py --registry thesis \
    --shard-dir runs/sih2024/shards \
    --n-jobs "${SLURM_CPUS_PER_TASK}" \
    ${UNIT_ARGS}
