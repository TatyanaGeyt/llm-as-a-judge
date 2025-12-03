#!/usr/bin/env bash

# Скрипт-обёртка, позволяющий запускать set_env.py как команду "set_env".

# Путь к корню репозитория (каталог, где лежит этот скрипт)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Запускаем python-скрипт с переданными аргументами
python "${REPO_DIR}/set_env.py" "$@"
