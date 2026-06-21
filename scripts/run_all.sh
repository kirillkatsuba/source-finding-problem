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
DATA_URL=${DATA_URL:-"https://disk.360.yandex.ru/d/CmNttY9n4c3EdA"}

# Тянем data.tar.gz с публичной ссылки Yandex Disk, если данных еще нет.
fetch_data() {
    if ls data/pollution/*.nc >/dev/null 2>&1; then
        echo "data: already present, skip download"
        return
    fi
    local api="https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=${DATA_URL}"
    local href
    href=$(curl -sL "$api" | python3 -c "import sys, json; print(json.load(sys.stdin)['href'])") || true
    if [ -z "$href" ]; then
        echo "data: failed to resolve download url" >&2
        exit 1
    fi
    echo "data: downloading archive"
    curl -L -o data.tar.gz "$href"
    rm -rf _unpack && mkdir -p data _unpack
    tar -xzf data.tar.gz -C _unpack
    # архив может быть как 'pollution/ wind/', так и 'data/...'
    local pol win
    pol=$(find _unpack -type d -name pollution | head -1)
    win=$(find _unpack -type d -name wind | head -1)
    if [ -z "$pol" ] || [ -z "$win" ]; then
        echo "data: pollution/ or wind/ not found in archive" >&2
        exit 1
    fi
    rm -rf data/pollution data/wind
    mv "$pol" data/pollution
    mv "$win" data/wind
    rm -rf _unpack data.tar.gz
    echo "data: pollution=$(ls data/pollution/*.nc | wc -l) wind=$(ls data/wind/*.nc | wc -l)"
}

fetch_data
poetry run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo ">>> exp_000 trivial"
poetry run python experiments/exp_000_trivial/run.py --dataset both

echo ">>> exp_001 physical baseline (sakhalin)"
poetry run python experiments/exp_001_baseline/run_unified.py

echo ">>> exp_002 Transolver field-only (nsk)"
poetry run python experiments/exp_002_transolver/train_transolver.py \
    --dataset nsk --epochs "$EPOCHS" --batch-size "$BS" --lr "$LR" \
    --augment --normalize $EXTRA_FLAGS

echo ">>> exp_003 Transolver + heatmap (nsk)"
poetry run python experiments/exp_003_transolver_heatmap_multitask/train.py \
    --dataset nsk --epochs "$EPOCHS" --batch-size "$BS" --lr "$LR" \
    --augment --normalize $EXTRA_FLAGS

echo ">>> exp_004 Transolver + regressor (nsk)"
poetry run python experiments/exp_004_transolver_regressor_multitask/train.py \
    --dataset nsk --epochs "$EPOCHS" --batch-size "$BS" --lr "$LR" \
    --augment --normalize $EXTRA_FLAGS

echo ">>> exp_005 UNet (nsk)"
poetry run python experiments/exp_005_unet_baseline/train.py \
    --dataset nsk --epochs "$EPOCHS" --batch-size "$BS" --lr "$LR" \
    --base 32 --augment --normalize $EXTRA_FLAGS

echo ">>> exp_006 Transolver + heatmap, no wind (sakhalin)"
poetry run python experiments/exp_006_transolver_with_wind/train.py \
    --epochs "$EPOCHS" --batch-size 4 --lr "$LR" \
    --augment --rot90 --normalize \
    --name "sakhalin_no_wind" --out-suffix "no_wind" $EXTRA_FLAGS

echo ">>> exp_006 Transolver + heatmap + wind (sakhalin)"
poetry run python experiments/exp_006_transolver_with_wind/train.py \
    --epochs "$EPOCHS" --batch-size 4 --lr "$LR" \
    --include-wind --augment --rot90 --normalize \
    --name "sakhalin_with_wind" --out-suffix "with_wind" $EXTRA_FLAGS

echo ">>> exp_007 PINN (sakhalin, advection-diffusion loss)"
poetry run python experiments/exp_007_pinn/train.py \
    --epochs "$EPOCHS" --batch-size 4 --lr "$LR" \
    --augment --rot90 --normalize --w-physics 0.1 $EXTRA_FLAGS

echo ">>> aggregate + plots"
poetry run python experiments/aggregate.py
poetry run python experiments/plots.py
poetry run python tools/compare.py
