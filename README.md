# llm-as-a-judge

### Запуск:

Для работы скриптов:

```
chmod +x set_env.sh
chmod +x compute-criterias.sh
chmod +x llm-as-a-judge.sh
chmod +x ratings.sh
```

### Этап 1

Необходимо загрузить `api-base` и `api-key`. Для этого используется скрипт `set-env.sh`:
```
./set_env.sh --api-base ... --api-key ...
```

### Этап 2

При необходимости запустить оценку ответа по соответствию критериям. Для этого используется скрипт `compute-criterias.sh`:
```
./compute-criteria.sh
```

### Этап 3

Сравнение ответов у llm-as-a-jusge. Считается, что критерии, если они нужны, **уже посчитанны** на этапе 2. Для этого используется скрипт `llm-as-a-judge.sh`, аргумент - `prompt`:
```
./llm-as-a-judge.sh --prompt '...'
```

**Список доступных промптов:**
1. `default_criteria_prompt` - для критериев (этап 2)
2. `only_crit` - оценка только по критериям
3. `answers_and_crit` - оценка и по ответам, и по критериям
4. `answers_extracting_crit` - оценка по ответам, определяя соответствие критериям
5. `only_answers` - сравнение ответов без указания критериев

Добавить промпт можно в файл `artifacts/prompts/prompts.py` по аналогии с уже добавленными.

**user mode**: пользователь может добавить свой собственный `prompt`. для этого необходимо создать соответстующую функцию в `artifacts/prompts/user_prompts.py`. 

### Этап 4

Получение elo-рейтингов.
```
./ratings.sh --prompt '...'
```
Рейтинги появляются в `artifacts/judge_evaluation_results/{prompt}/ratings.json` и `artifacts/judge_evaluation_results/{prompt}/ratings.csv`
