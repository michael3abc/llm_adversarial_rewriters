#!/usr/bin/env bash
set -euo pipefail

# Google Drive file ID and target locations.
FILE_ID="1Gmt4HpmIEBy32qW85BCUqAEUwW1xElAo"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${ROOT_DIR}"
ARCHIVE_PATH="${ROOT_DIR}/download.zip"

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown 未安裝，請先執行：pip install gdown" >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "系統缺少 unzip，請安裝 unzip 後再執行此腳本。" >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"

if [ ! -f "${ARCHIVE_PATH}" ]; then
  echo "下載模型檔案到 ${ARCHIVE_PATH}..."
  gdown --id "${FILE_ID}" -O "${ARCHIVE_PATH}"
else
  echo "偵測到已存在 ${ARCHIVE_PATH}，略過下載。"
fi

echo "解壓縮模型到 ${DEST_DIR}..."
unzip -o "${ARCHIVE_PATH}" -d "${DEST_DIR}"
echo "完成。"
