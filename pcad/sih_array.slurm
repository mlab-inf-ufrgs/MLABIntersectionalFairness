#!/bin/bash
# Etapa 1 dos experimentos do SIH 2024 no PCAD: modelos de referência.
#   RandomForest e XGBoost (5 métricas de otimização cada) e FairMLP com λ fixo,
#   17 capítulos x 3 dobras x 5 sementes = 3.825 unidades (uma por linha de pcad/units_sih.txt).
#
# O PCAD limita o número de jobs por usuário na fila (QOSMaxSubmitJobPerUserLimit), e cada
# elemento de array conta como um job. Por isso cada job do array pega uma FATIA das unidades
# (linhas t, t+N, t+2N, ..., com N = tamanho do array) e roda PARALLEL unidades ao mesmo
# tempo, cada uma com 4 núcleos. Unidades já concluídas são puladas: se um job cair, basta
# reenviar o mesmo array.
#
# Preparação (uma vez, na raiz do repositório, no PCAD):
#   python3 -m venv ~/venvs/tese && source ~/venvs/tese/bin/activate
#   pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
#   pip install -r requirements-experiments.txt
#
# Envio:
#   SIH=$(python -c "from utils.thesis_datasets import THESIS_DATASETS as T; print(' '.join(k for k in T if k.startswith('sih_v2_2024')))")
#   python scripts/run_experiments.py --registry thesis --print-units --datasets $SIH \
#       --seeds 42 43 44 45 46 > pcad/units_sih.txt
#   mkdir -p runs/sih2024/logs        # o Slurm não cria a pasta dos logs
#   sbatch --array=1-10 pcad/sih_array.slurm     # N jobs, abaixo do limite de jobs por usuário
#
# Acompanhar / juntar:
#   ls runs/sih2024/shards/*.metrics.parquet | wc -l      # meta: 3825
#   python scripts/collect_results.py --shard-dir runs/sih2024/shards --out-dir runs/sih2024/results
#
# Tempo medido (4 núcleos, maior capítulo): RandomForest 8 min, XGBoost 3,5 min, FairMLP 1 min
# por unidade; ~270 h de unidade no total. Com 10 jobs x 4 unidades em paralelo: ~7 h.

#SBATCH --job-name=sih2024
#SBATCH --partition=hype,tupi,shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/sih2024/logs/%A_%a.out
#SBATCH --error=runs/sih2024/logs/%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/venvs/tese/bin/activate

UNITS=pcad/units_sih.txt
THREADS=4                                              # núcleos por unidade
PARALLEL=$(( SLURM_CPUS_PER_TASK / THREADS ))          # unidades simultâneas no job
TOTAL=$(wc -l < "${UNITS}")
N=${SLURM_ARRAY_TASK_COUNT:-1}
T=$(( SLURM_ARRAY_TASK_ID - ${SLURM_ARRAY_TASK_MIN:-1} + 1 ))

echo "[job ${T}/${N}] linhas ${T}, $((T + N)), ... de ${TOTAL}; ${PARALLEL} em paralelo @ $(hostname) $(date -Is)"

run_line() {
    local line=$1
    local args
    args=$(sed -n "${line}p" "${UNITS}")
    python scripts/run_experiments.py --registry thesis \
        --shard-dir runs/sih2024/shards --n-jobs "${THREADS}" ${args} \
        > "runs/sih2024/logs/unit_${line}.log" 2>&1 \
        && echo "[unit ${line}] ok $(date -Is)" \
        || echo "[unit ${line}] ERRO (ver runs/sih2024/logs/unit_${line}.log) $(date -Is)"
}
export -f run_line
export UNITS THREADS

seq "${T}" "${N}" "${TOTAL}" | xargs -P "${PARALLEL}" -I{} bash -c 'run_line {}'

echo "[job ${T}/${N}] fim $(date -Is)"
