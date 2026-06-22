#!/usr/bin/env bash
# Скачать data.tar.gz с публичной ссылки Yandex Disk в data/pollution и data/wind.
# Run:    bash scripts/fetch_data.sh
# Другая ссылка: DATA_URL="https://disk.360.yandex.ru/d/..." bash scripts/fetch_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_URL=${DATA_URL:-"https://disk.360.yandex.ru/d/CmNttY9n4c3EdA"}

if ls data/pollution/*.nc >/dev/null 2>&1; then
    echo "data: already present, skip download"
    exit 0
fi

api="https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=${DATA_URL}"
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
# macOS AppleDouble (._*) из tar ломают glob xarray -> чистим
find data/pollution data/wind -name '._*' -delete 2>/dev/null || true
echo "data: pollution=$(ls data/pollution/*.nc | wc -l) wind=$(ls data/wind/*.nc | wc -l)"
