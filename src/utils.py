import json
import logging
from pathlib import Path
import subprocess
import os


def get_repo_path():
    """
    Вычисляет путь к корню репозитория.
    Возвращает директорию, где находится этот скрипт или git root.
    """
    script_dir = Path(__file__).parent.absolute()
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=script_dir,
            capture_output=True,
            text=True,
            check=True
        )
        return Path(result.stdout.strip()).resolve()
    except Exception:
        return script_dir


def write_json(filename, data, log):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('[\n')
            end = len(data)
            for i, d in enumerate(data):
                f.write('    ')
                json.dump(d, f, ensure_ascii=False)
                if i != end - 1:
                    f.write(',')
                f.write('\n')
            f.write(']\n')
    except Exception as e:
        log.error(f'[write_json] {e}')


def get_data_from_dir(path, log):
    model_dir = Path(path)
    pattern = r'*.json'
    files = list(model_dir.glob(pattern))
    if not files:
        log.error(f'No model files in directory {model_dir}')
        raise FileNotFoundError

    data = {}
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                cur_data = json.load(f)
                for i, d in enumerate(cur_data):
                    d['id'] = i
                model_name = file.stem
                data[model_name] = cur_data
        except Exception as e:
            log.error(f'Error in loading file {file}: {e}')
    log.info(f'Loaded files for models {list(data.keys())}')
    return data


def cross_battle_join(
    model_data, 
    model_crit,
    ref_data,
    ref_crit
):
    final_data = {}
    for m_name, m_answers in model_data.items():
        final_data[m_name] = {}
        for r_name, r_answers in ref_data.items():
            final_data[m_name][r_name] = []
            for idx, m_answer in enumerate(m_answers):
                entry = {
                    "id": idx,
                    "instruction": m_answer["instruction"],
                    "model": m_name,
                    "r": r_name,
                    "model_output": m_answer["output"],
                    "model_scores": model_crit[m_name][idx]['output'],
                    "reference_output": r_answers[idx]["output"],
                    "reference_scores": ref_crit[r_name][idx]['output'],
                    'reverse': False
                }
                rev_entry = {
                    "id": idx,
                    "instruction": m_answer["instruction"],
                    "model": r_name,
                    "r": m_name,
                    "model_output": r_answers[idx]["output"],
                    "model_scores": ref_crit[r_name][idx]['output'],
                    "reference_output": m_answer["output"],
                    "reference_scores": model_crit[m_name][idx]['output'],
                    'reverse': True
                }
                final_data[m_name][r_name].append(entry)
                final_data[m_name][r_name].append(rev_entry)
    return final_data


# def setup_logger(log_name=''):
#     log_dir = Path(f"{os.environ['REPO_PATH']}/logs")
#     log_dir.mkdir(parents=True, exist_ok=True)

#     if not log_name:
#         log_name = Path(__file__).stem

#     log_file = log_dir / f"{log_name}.log"
#     log_file_dir = log_file.parent
#     log_file_dir.mkdir(parents=True, exist_ok=True)

#     logging.basicConfig(
#         filename=str(log_file),
#         level=logging.INFO,
#         format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
#     )

#     log = logging.getLogger(log_name)
#     log.info('----------------------------------------------------------------------------------------')

#     return log


def setup_logger(log_name: str):
    log_dir = Path(os.environ["REPO_PATH"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{log_name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # создаём логер
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    # если handler'ы уже есть — не добавляем снова (иначе дублирование строк)
    if not logger.handlers:
        handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False  # чтобы не уходило в root-logger
    logger.info('-' * 88)

    return logger



def get_model_name(sample):
    if 'generator' in sample:
        return sample['generator']
    if 'model' in sample:
        return sample['model']
    if 'r' in sample:
        return sample['r']
    raise ValueError(f"Can not get model name from sample {sample}")


def get_coords(sample):
    coords = {}
    if 'id' in sample:
        coords['id'] = sample['id']
    coords['m'] = get_model_name(sample)
    if 'r' in sample:
        coords['r'] = sample['r']
    return coords