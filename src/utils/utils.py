import os
import yaml
import logging
import shutil
import pandas as pd
from datetime import datetime
import itertools


DEFAULT_LLM_CACHE_DIR = os.getenv('CACHE_PATH', '/tmp/huggingface_cache')


class ExcludeLiteLLMWarningFilter(logging.Filter):
    def filter(self, record):
        # Exclude messages containing "specific warning"
        return "Only some together models support function calling/response_format." not in record.getMessage()


def setup_logging(experiment_dir, debug=False):
    # General logging configuration
    log_level = logging.DEBUG if debug else logging.INFO
    general_log_file = os.path.join(experiment_dir, 'experiment.log')
    lite_llm_log_file = os.path.join(experiment_dir, 'litellm.log')

    file_handler = logging.FileHandler(general_log_file, mode='a')
    stream_handler = logging.StreamHandler()
    file_handler.addFilter(ExcludeLiteLLMWarningFilter())
    stream_handler.addFilter(ExcludeLiteLLMWarningFilter())
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            file_handler,
            stream_handler
        ]
    )

    # Specific logger for LiteLLM
    litellm_logger = logging.getLogger('LiteLLM')
    litellm_handler = logging.FileHandler(lite_llm_log_file, mode='a')
    litellm_handler.setLevel(logging.DEBUG)
    litellm_formatter = logging.Formatter('%(asctime)s,%(msecs)d %(levelname)s %(name)s: %(message)s', '%H:%M:%S')
    litellm_handler.setFormatter(litellm_formatter)
    litellm_handler.addFilter(ExcludeLiteLLMWarningFilter())
    litellm_logger.addHandler(litellm_handler)
    litellm_logger.propagate = False


def find_git_root(path):
    if os.path.exists(os.path.join(path, '.git')):
        return path
    else:
        parent, _ = os.path.split(path)
        if parent == '':
            return None
        else:
            return find_git_root(parent)


def load_existing_answers(answers_file_path):
    try:
        if os.path.exists(answers_file_path):
            return pd.read_csv(answers_file_path)
        else:
            return pd.DataFrame()  # Return an empty DataFrame if the file does not exist
    except Exception as e:
        logging.error(f"Error loading existing answers: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def create_experiment_directory(base_dir, experiment_name):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_dir = os.path.join(base_dir, f"{timestamp}_{experiment_name}")
    os.umask(0)  # see https://stackoverflow.com/questions/47618490/python-create-a-directory-with-777-permissions
    os.makedirs(experiment_dir, mode=0o777, exist_ok=True)
    return experiment_dir


def get_questions_path(experiment_dir):
    return os.path.join(experiment_dir, 'questions.csv')


def get_answers_path(experiment_dir):
    return os.path.join(experiment_dir, 'answers.csv')


def is_directory_empty(directory):
    """Check whether a directory is empty."""
    return not os.listdir(directory)

def setup_experiment_files(base_dir, source_question_file):
    # Extract the basename without extension for use as the experiment name
    experiment_name = os.path.splitext(os.path.basename(source_question_file))[0]
    experiment_dir = create_experiment_directory(base_dir, experiment_name)
    shutil.copy(source_question_file, get_questions_path(experiment_dir))
    source_question_settings_file = source_question_file.replace('.csv', '_settings.csv')
    if os.path.exists(source_question_settings_file):
        shutil.copy(source_question_settings_file, os.path.join(experiment_dir, 'questions_settings.csv'))
    return experiment_dir


def product_of_dict_grid(dict_grid: dict):
    """
    For a dict where each value is a list, return a list where each element is a dictionary with the same keys as the
    input dict, and each value is a different combination of the input lists. This is a Cartesian product.
    """
    keys, values = zip(*dict_grid.items())
    results = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return results


def independent_variations_of_dict_grid(dict_grid: dict):
    """
    For a dict where each value is a list, return a list of dicts that are each a variant of that dict where all values
    are set to the first value in each dict_grid list except for the value of one key, which is set to a different value
    from its list of values in dict_grid. Finally, a dict with all values set to the first value in each dict_grid list
    is prepended to the list of results.
    """
    base_dict = {key: dict_grid[key][0] for key in dict_grid}

    results = []
    for key in base_dict:
        for value in dict_grid[key][1:]:
            new_dict = base_dict.copy()
            new_dict[key] = value
            results.append(new_dict)
    results = [base_dict.copy()] + results
    return results


def multilingual(dict_grid: dict):
    variation_lengths = [len(v) for v in dict_grid.values()]
    assert all(v == variation_lengths[0] for v in variation_lengths), \
        "All values in the dict_grid must have the same length (translations of the same prompt"
    nb_languages = variation_lengths[0]

    assert 'language' in dict_grid, "The dict_grid must contain a 'language' key"

    results = []
    for i in range(nb_languages):
        language_dict = {key: dict_grid[key][i] for key in dict_grid}
        language_dict['language_idx'] = i
        results.append(language_dict)
    return results