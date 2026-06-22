#!/usr/bin/env bash
# Скачать данные и прогнать все эксперименты по очереди (vast.ai).
# Setup: pip install poetry && poetry install --no-root
# Run:   EPOCHS=100 BS=8 LR=1e-3 bash scripts/run_all.sh --comet
# Smoke: bash scripts/run_all.sh --smoke
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA_FLAGS="$@"
EPOCHS=${EPOCHS:-100}
BS=${BS:-8}
LR=${LR:-1e-3}
# по умолчанию file-split + трансляция (случайный сдвиг источника каждую эпоху)
# для source-disjoint split: SPLIT_FLAGS=--source-split bash scripts/run_all.sh
SPLIT_FLAGS=${SPLIT_FLAGS:-}
TRANSLATE_FLAGS=${TRANSLATE_FLAGS:---translate}

bash scripts/fetch_data.sh
poetry run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo ">>> exp_000 trivial"
poetry run python experiments/exp_000_trivial/run.py --dataset both $SPLIT_FLAGS

echo ">>> exp_001 physical baseline (sakhalin)"
poetry run python experiments/exp_001_baseline/run_unified.py $SPLIT_FLAGS

echo ">>> exp_002 Transolver field-only (nsk)"
poetry run python experiments/exp_002_transolver/train_transolver.py \
    --dataset nsk --epochs "$EPOCHS" --batch-size "$BS" --lr "$LR" \
    --augment --normalize $SPLIT_FLAGS $TRANSLATE_FLAGS $EXTRA_FLAGS

echo ">>> exp_003 Transolver + heatmap (nsk)"
poetry run python experiments/exp_003_transolver_heatmap_multitask/train.py \
    --dataset nsk --epochs "$EPOCHS" --batch-size "$BS" --lr "$LR" \
    --augment --normalize $SPLIT_FLAGS $TRANSLATE_FLAGS $EXTRA_FLAGS

echo ">>> exp_004 Transolver + regressor (nsk)"
poetry run python experiments/exp_004_transolver_regressor_multitask/train.py \
    --dataset nsk --epochs "$EPOCHS" --batch-size "$BS" --lr "$LR" \
    --augment --normalize $SPLIT_FLAGS $TRANSLATE_FLAGS $EXTRA_FLAGS

echo ">>> exp_005 UNet (nsk)"
poetry run python experiments/exp_005_unet_baseline/train.py \
    --dataset nsk --epochs "$EPOCHS" --batch-size "$BS" --lr "$LR" \
    --base 32 --augment --normalize $SPLIT_FLAGS $TRANSLATE_FLAGS $EXTRA_FLAGS

echo ">>> exp_006 Transolver + heatmap, no wind (sakhalin)"
poetry run python experiments/exp_006_transolver_with_wind/train.py \
    --epochs "$EPOCHS" --batch-size 4 --lr "$LR" \
    --augment --rot90 --normalize \
    --name "sakhalin_no_wind" --out-suffix "no_wind" $SPLIT_FLAGS $TRANSLATE_FLAGS $EXTRA_FLAGS

echo ">>> exp_006 Transolver + heatmap + wind (sakhalin)"
poetry run python experiments/exp_006_transolver_with_wind/train.py \
    --epochs "$EPOCHS" --batch-size 4 --lr "$LR" \
    --include-wind --wind-per-frame --augment --rot90 --normalize \
    --name "sakhalin_with_wind" --out-suffix "with_wind" $SPLIT_FLAGS $TRANSLATE_FLAGS $EXTRA_FLAGS

echo ">>> exp_007 PINN (sakhalin, advection-diffusion loss)"
poetry run python experiments/exp_007_pinn/train.py \
    --epochs "$EPOCHS" --batch-size 4 --lr "$LR" \
    --augment --rot90 --normalize --w-physics 0.1 $SPLIT_FLAGS $TRANSLATE_FLAGS $EXTRA_FLAGS

echo ">>> aggregate + plots"
poetry run python experiments/aggregate.py
poetry run python experiments/plots.py
poetry run python tools/compare.py
