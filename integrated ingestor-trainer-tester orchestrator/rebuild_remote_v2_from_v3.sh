#!/usr/bin/env bash
set -u

# Rebuild remote v2 combined dataset from remote v3 combined dataset.
# This is intended to be run directly from your terminal (not via short-lived tool sessions),
# because SMB/GVFS copies can take a long time.

BASE="/run/user/1001/gvfs/smb-share:server=cu-genvpn-comp-10.180.19.223.int.colorado.edu,share=dl%20project%20ssd/integrated prototype data/patch training datasets and pipeline validation data/Integrated_prototype_datasets/integrated pipeline datasets"
SRC="${BASE}/v3_20260212_union_forresti_frontalis_tremulans_photinus-knulli"
DST="${BASE}/v2_20260204_union_forresti_frontalis_tremulans"
TMP_INCOMPLETE="${BASE}/__incomplete_v3_20260212_union_forresti_frontalis_tremulans_photinus-knulli"
LOG="${1:-.rebuild_v2_from_v3_remote.log}"

if [[ ! -d "${SRC}" ]]; then
  echo "[rebuild-v2] ERROR: source not found: ${SRC}"
  exit 1
fi

mkdir -p "${DST}"
echo "[rebuild-v2] $(date '+%F %T') start" | tee -a "${LOG}"
echo "[rebuild-v2] src=${SRC}" | tee -a "${LOG}"
echo "[rebuild-v2] dst=${DST}" | tee -a "${LOG}"
echo "[rebuild-v2] log=${LOG}" | tee -a "${LOG}"

cp -u "${SRC}/.DS_Store" "${DST}/.DS_Store" 2>/dev/null || true
cp -u "${SRC}/import_manifest.json" "${DST}/import_manifest.json" 2>/dev/null || true

pass=1
max_passes=200
CP_PASS_TIMEOUT_SECONDS="${CP_PASS_TIMEOUT_SECONDS:-900}"
while [[ "${pass}" -le "${max_passes}" ]]; do
  echo "[rebuild-v2] $(date '+%F %T') pass=${pass} begin timeout=${CP_PASS_TIMEOUT_SECONDS}s" | tee -a "${LOG}"
  timeout --foreground "${CP_PASS_TIMEOUT_SECONDS}" cp -ru "${SRC}/." "${DST}/" >> "${LOG}" 2>&1
  rc=$?
  echo "[rebuild-v2] $(date '+%F %T') pass=${pass} rc=${rc}" | tee -a "${LOG}"

  if [[ "${rc}" -eq 0 ]]; then
    echo "[rebuild-v2] $(date '+%F %T') copy complete" | tee -a "${LOG}"
    rm -rf "${TMP_INCOMPLETE}" >> "${LOG}" 2>&1 || true
    echo "[rebuild-v2] $(date '+%F %T') temp cleanup attempted" | tee -a "${LOG}"
    exit 0
  fi
  if [[ "${rc}" -eq 124 ]]; then
    echo "[rebuild-v2] $(date '+%F %T') pass=${pass} timed out; retrying" | tee -a "${LOG}"
  fi

  pass=$((pass + 1))
  sleep 10
done

echo "[rebuild-v2] $(date '+%F %T') gave up after ${max_passes} passes" | tee -a "${LOG}"
exit 2
