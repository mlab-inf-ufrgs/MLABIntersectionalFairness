#!/bin/bash
# Etapa 1 dos experimentos do SIH 2024 no PCAD: modelos de referência.
#   RandomForest e XGBoost (5 métricas de otimização cada) e FairMLP com λ fixo,
#   17 capítulos x 3 dobras x 5 sementes = 3.825 unidades.
# Um job do array = uma unidade (capítulo x modelo x configuração x dobra x semente).
# Unidades já concluídas são puladas, então o array pode ser reenviado sem refazer nada.
#
# Preparação (uma vez, na raiz do repositório, no PCAD):
#   python -m venv ~/venvs/tese && source ~/venvs/tese/bin/activate
#   pip install -r requirements-experiments.txt
#   # copiar data/processed/thesis/sih_v2/2024/ para o PCAD (os parquets estão no git)
#
# Envio:
#   SIH=$(python -c "from utils.thesis_datasets import THESIS_DATASETS as T; print(' '.join(k for k in T if k.startswith('sih_v2_2024')))")
#   python scripts/run_experiments.py --registry thesis --print-units --datasets $SIH \
#       --seeds 42 43 44 45 46 > pcad/units_sih.txt
#   mkdir -p runs/sih2024/logs        # o Slurm não cria a pasta dos logs
#   # o índice do array vai no máximo até 1000 no PCAD: 4 blocos com deslocamento (3.825 linhas)
#   sbatch --array=1-1000%40 --export=ALL,OFFSET=0    pcad/sih_array.slurm
#   sbatch --array=1-1000%40 --export=ALL,OFFSET=1000 pcad/sih_array.slurm
#   sbatch --array=1-1000%40 --export=ALL,OFFSET=2000 pcad/sih_array.slurm
#   sbatch --array=1-825%40  --export=ALL,OFFSET=3000 pcad/sih_array.slurm
#
# Ao final (ou a qualquer momento, para ver o parcial):
#   python scripts/collect_results.py --shard-dir runs/sih2024/shards --out-dir runs/sih2024/results
#
# CONFERIR com a documentação do PCAD: limite de tempo das partições, tamanho máximo de
# array (se N passar do limite, dividir em --array=1-1000, 1001-2000, ...) e se o venv
# acima funciona ou se é preciso carregar um módulo de Python antes.

#SBATCH --job-name=sih2024
#SBATCH --partition=hype,tupi,shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Medido localmente (4 núcleos, maior capítulo, 200 mil linhas): RandomForest 8 min,
# XGBoost 3,5 min, FairMLP 1 min por unidade. 2 h dá folga para nós mais lentos e
# um limite curto ajuda o Slurm a encaixar os jobs nas brechas da fila.
#SBATCH --time=02:00:00
#SBATCH --output=runs/sih2024/logs/%A_%a.out
#SBATCH --error=runs/sih2024/logs/%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/venvs/tese/bin/activate

# O PCAD limita o índice do array a 1000 (MaxArraySize = 1001): blocos seguintes usam OFFSET.
LINE=$(( SLURM_ARRAY_TASK_ID + ${OFFSET:-0} ))
UNIT_ARGS=$(sed -n "${LINE}p" pcad/units_sih.txt)
if [ -z "${UNIT_ARGS}" ]; then echo "[unit ${LINE}] linha vazia, nada a fazer"; exit 0; fi
echo "[unit ${LINE}] ${UNIT_ARGS} @ $(hostname) $(date -Is)"

python scripts/run_experiments.py --registry thesis \
    --shard-dir runs/sih2024/shards \
    --n-jobs "${SLURM_CPUS_PER_TASK}" \
    ${UNIT_ARGS}

echo "[unit ${LINE}] fim $(date -Is)"
