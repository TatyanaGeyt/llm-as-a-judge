from pathlib import Path
import json
import logging


def get_json_files(path):
    model_dir = Path(path)
    pattern = r'*.json'
    files = list(model_dir.glob(pattern))
    assert files, f'No model files in directory {model_dir}'
    return files


def get_data(file):
    if not file:
        return []
    with open(file, 'r', encoding='utf-8') as f:
        cur_data = json.load(f)
        for i, d in enumerate(cur_data):
            d['id'] = i
        return cur_data
    

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


def setup_logger(log_dir: str, log_name: str):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{log_name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    logger.info('-' * 88)

    return logger


def write_json(filename, data, log):
    filename = filename.resolve()
    try:
        if isinstance(data, list):
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
        elif isinstance(data, dict):
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f'[write_json] {e}')


def combine_sample(m_sample, b_sample):
    combined = {}
    all_keys = set(m_sample.keys()) | set(b_sample.keys())
    for key in all_keys:
        val_m = m_sample.get(key, default=None)
        val_b = b_sample.get(key, default=None)
        if val_m is None and val_b is None:
            continue
        if val_m is None and val_b is not None:
            combined[key] = val_b
        elif val_b is None and val_m is not None:
            combined[key] = val_m
        elif val_m == val_b:
            combined[key] = val_m
        else:
            combined[f"model_{key}"] = val_m
            combined[f"reference_{key}"] = val_b
    return combined


def make_battle(data, reference_data):
    ref_lookup = {sample.get('id'): sample for sample in reference_data if 'id' in sample}
    battles = []
    for m_sample in data:
        m_id = m_sample.get('id')
        if m_id in ref_lookup:
            b_sample = ref_lookup[m_id]
            merged = combine_sample(m_sample, b_sample)
            battles.append(merged)
    return battles


def check_requests(
    sent_batch, 
    received_batch,
    batch_id,
    judge, 
    attempt=3
):
    restart = True
    iter = 0
    while restart and iter < attempt:
        judge.log.info(f'[RESTART] iteration {iter}')
        restart = False
        restart_id = [
            d['id'] for d in received_batch
            if 'error' in d.keys()
        ]
        if not restart_id:
            break
        
        good_tasks = [
            d for d in received_batch
            if 'error' not in d.keys()
        ]
        restart = True
        judge.log.info(f'[RESTART] {batch_id}: {restart_id}')
        
        restart_data = [
            d for d in sent_batch
            if d['id'] in restart_id
        ]
        good_tasks += judge.generate_batch(
            samples_batch=restart_data,
            batch_id=batch_id
        )
        received_batch = sorted(good_tasks, key=lambda x: x['id'])
        iter += 1
    
    assert restart == False, f'[RESTART] no more attempts, some errors accrue in batch {batch_id}'
    judge.log.info(f'[RESTART] no more errors')
    return received_batch