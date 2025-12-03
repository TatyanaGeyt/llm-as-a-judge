#!/usr/bin/env bash

# Скрипт загружает переменные из .env и запускает python src/ev_criteria.py

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Загружаем .env
if [[ -f "${REPO_DIR}/.env" ]]; then
    set -o allexport
    source "${REPO_DIR}/.env"
    set +o allexport
else
    echo "Файл .env не найден в ${REPO_DIR}"
    exit 1
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Запуск python
python "${REPO_DIR}/src/ev_criteria.py"
