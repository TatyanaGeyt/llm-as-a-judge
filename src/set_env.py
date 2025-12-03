# set_env.py - скрипт для установки переменных окружения

import argparse
from src.utils import get_repo_path


def create_env_file(api_base, api_key, repo_path: str=''):
    """Создает или обновляет .env файл"""
    if not repo_path:
        repo_path = get_repo_path()
    env_content = f"""# Файл окружения, созданный set.py

REPO_PATH={repo_path}
API_BASE={api_base}
OPENAI_API_KEY={api_key}
"""
    with open(f'{repo_path}/.env', 'w', encoding='utf-8') as f:
        f.write(env_content)
    return repo_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setting environment variables for a project')
    
    parser.add_argument(
        '--api-base',
        required=True,
        help='Base URL API'
    )
    parser.add_argument(
        '--api-key',
        required=True,
        help='API OpenAI key'
    )
    args = parser.parse_args()

    repo_path = get_repo_path()
    create_env_file(args.api_base, args.api_key, repo_path=repo_path)
