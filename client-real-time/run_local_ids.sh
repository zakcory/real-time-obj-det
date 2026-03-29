#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/client/secrets/config.yaml"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}"
  exit 1
fi

if [[ -x "${SCRIPT_DIR}/run_local" ]]; then
  RUN_LOCAL_CMD="${SCRIPT_DIR}/run_local"
elif [[ -x "${SCRIPT_DIR}/run_local.sh" ]]; then
  RUN_LOCAL_CMD="${SCRIPT_DIR}/run_local.sh"
else
  echo "Could not find an executable run_local script in ${SCRIPT_DIR}"
  exit 1
fi

if ! grep -Eq '^[[:space:]]*ids:[[:space:]]*\[[^]]*\][[:space:]]*$' "${CONFIG_FILE}"; then
# For a YAML layout like:
# ids:
#   - "1"
#   - "2"
# use this instead:
# if ! grep -Pzoq '^[[:space:]]*ids:[[:space:]]*\n([[:space:]]*-[[:space:]]*".*"[[:space:]]*\n?)+' "${CONFIG_FILE}"; then
  echo "Could not find an ids list line in ${CONFIG_FILE}"
  exit 1
fi

CONFIG_BAK="$(mktemp)"
cp "${CONFIG_FILE}" "${CONFIG_BAK}"

restore_config() {
  cp "${CONFIG_BAK}" "${CONFIG_FILE}"
  rm -f "${CONFIG_BAK}"
}
trap restore_config EXIT

for end_id in 1 2 3 4 5; do
  ids_csv="$(seq -s ', ' 1 "${end_id}")"
  ids_line="  ids: [${ids_csv}]"
  # For the indented YAML-list layout above, build the replacement block instead:
  # ids_line="$(printf '  ids:\n')"
  # for id in $(seq 1 "${end_id}"); do
  #   ids_line+=$(printf '    - "%s"\n' "${id}")
  # done

  sed -E -i "0,/^[[:space:]]*ids:[[:space:]]*\\[[^]]*\\][[:space:]]*$/s//${ids_line}/" "${CONFIG_FILE}"
  # For the indented YAML-list layout above, use this instead:
  # sed -Ez -i "0,/^[[:space:]]*ids:[[:space:]]*\n([[:space:]]*-[[:space:]]*\".*\"[[:space:]]*\n?)+/s//${ids_line}/" "${CONFIG_FILE}"

  echo "Running with ids: [${ids_csv}]"
  "${RUN_LOCAL_CMD}"
  status=$?

  if [[ ${status} -ne 0 && ${status} -ne 130 && ${status} -ne 143 ]]; then
    echo "run_local exited with status ${status}. Stopping."
    exit "${status}"
  fi
done

echo "Completed all id sets."
