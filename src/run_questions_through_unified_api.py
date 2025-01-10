import os
import yaml
import logging
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
import gc
import torch
import ast
import pandas as pd
import asyncio
import litellm

from llms import (LITE_LLM_LLMS, LANGCHAIN_LLMS, OPENAI_LLMS, LITE_LLM_METADATA_DICT, LOCAL_LLM_METADATA_DICT,
                  LangChainModel, LiteLLM_model, OpenAIModel, LOCAL_LLMS_CLASSES)
from utils import (setup_logging, load_existing_answers, load_config, setup_experiment_files, get_questions_path,
                   get_answers_path, DEFAULT_LLM_CACHE_DIR)
from utils.chat import ChatCompletor
from answer_extraction import generate_message

# Next two line are for running the script on a specific GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# Load environment variables from .env file first
load_dotenv()

# Then try to load API keys from specified path if it exists
API_KEY_PATH = os.getenv('API_KEYS_ENV')
if API_KEY_PATH and os.path.exists(API_KEY_PATH):
    load_dotenv(dotenv_path=API_KEY_PATH)
    logging.info(f"Loaded API keys from {API_KEY_PATH}")
else:
    logging.info("Using environment variables for API keys")

# Validate required API keys are present
REQUIRED_API_KEYS = [
    'OPENAI_API_KEY',
    'ANTHROPIC_API_KEY',
    'HUGGINGFACE_TOKEN',
    'MISTRAL_API_KEY',
    'TOGETHER_API_KEY',
    'PERPLEXITY_API_KEY',
    'GEMINI_API_KEY'
]

missing_keys = [key for key in REQUIRED_API_KEYS if not os.getenv(key)]
if missing_keys:
    logging.warning(f"Missing required API keys: {', '.join(missing_keys)}")

MODEL_LIST = LITE_LLM_LLMS + LANGCHAIN_LLMS + OPENAI_LLMS


def group_models_by_provider(model_list):
    models_by_provider = {}
    unassigned_models = []

    for model in model_list:
        # Check if it's an OpenAI-style model first
        if model in OPENAI_LLMS:
            raise NotImplementedError("OpenAI models are not supported in this script.")
            # openai_style_models.append(model)
        # Then handle litellm models
        elif "/" in model:
            provider, model_name = model.split('/', 1)
            if provider not in models_by_provider:
                models_by_provider[provider] = []
            models_by_provider[provider].append(model)
        # Only add to unassigned if it's in LANGCHAIN_LLMS
        elif model in LANGCHAIN_LLMS:
            raise NotImplementedError("LangChain models are not supported in this script.")
            # unassigned_models.append(model)

    return models_by_provider, unassigned_models


def wrap_question_as_message(question):
    return [{"content": question, "role": "user"}]


async def process_question(model, idx, row, answers_file_path, existing_answers_df):
    logging.debug(row)
    previous_response = ""
    messages = []

    question_processed = not existing_answers_df.loc[
        (existing_answers_df['question_idx'] == idx) & (existing_answers_df['model'] == model.name)
        ].empty

    if question_processed:
        # logging.info(f"Question {idx} for model {model.name} has already been processed. Skipping...")
        return

    result_row = {
        "question_idx": idx,
        "model": model.name,
        "total_cost_usd": 0,
        "stage_1": pd.NA,
        "stage_1_response": pd.NA,
        "stage_2": pd.NA,
        "stage_2_response": pd.NA,
        "stage_3": pd.NA,
        "stage_3_response": pd.NA,
        "final_response": pd.NA,
        "extracted": pd.NA,
    }
    for stage in (key for key in row.keys() if key.startswith('stage_')):
        question = row[stage]
        if isinstance(question, float):
            question = ''

        if previous_response is None:
            previous_response = ""
        question = question.replace("<ANS>", previous_response)

        if '<RESET>' in question:
            messages = []
            question = question.replace("<RESET>", '')

        if question == "":
            logging.info(f"Skipping stage {stage} message for question {idx} to model {model.name}, as it is empty.")
            continue

        result_row[stage] = question
        messages.extend(wrap_question_as_message(question))

        logging.info(f"Sending messages to model {model.name} for question {idx} stage {stage}: {messages}")
        response, total_cost_usd = await model.get_response(idx, stage, messages)

        if response:
            try:
                # Check if choices list is not empty before accessing
                if response['choices'] and len(response['choices']) > 0:
                    try:
                        response_str = response['choices'][0]['message']['content']
                    except (KeyError, AttributeError):
                        try:
                            response_str = response["choices"][0]["text"]
                        except (KeyError, AttributeError):
                            logging.error(f"Unexpected response format from model {model.name}: {response}")
                            response_str = ""
                else:
                    logging.error(f"Empty choices list in response from model {model.name}: {response}")
                    response_str = ""

                previous_response = response_str
                result_row[f"{stage}_response"] = response_str
                messages.extend([{"content": response_str, "role": "assistant"}])
                logging.info(f"Got response from model {model.name} for question {idx} stage {stage}: {response}")
            except Exception as e:
                logging.error(f"Error processing response from {model.name}: {e}")
                response_str = ""
        else:
            logging.error(f"Failed to get response for model {model.name}, question {idx}, stage {stage}")

    result_row["final_response"] = previous_response
    result_row["total_cost_usd"] = total_cost_usd

    df_row = pd.DataFrame([result_row])
    return df_row


# async def process_openai_style_models(models, questions_df, answers_file_path, existing_answers_df):
#     """Process models that use OpenAI-style APIs"""
#     for model_name in models:
#         if (existing_answers_df['model'] == model_name).sum() == len(questions_df):
#             logging.info(f"Model {model_name} has already processed all questions. Skipping...")
#             continue
#
#         # Configure based on the provider prefix
#         if model_name.startswith("xai/"):
#             model = OpenAIModel(
#                 model_name=model_name,
#                 api_key=os.getenv("XAI_API_KEY"),
#                 base_url="https://api.x.ai/v1"
#             )
#
#         for idx, row in questions_df.iterrows():
#             logging.info(f"Starting async process for question {idx} to {model_name}")
#             await process_question(model, idx, row, answers_file_path, existing_answers_df)


async def process_questions_for_provider(provider, models, questions_df, answers_file_path, existing_answers_df,
                                         batch_size=1):
    for model_name in models:

        try:
            supported_languages = LITE_LLM_METADATA_DICT[model_name]['supported_languages']
            logging.info(f"Supported languages for {model_name}: {supported_languages}. Only running those!")
            questions_df_supported = questions_df[questions_df['language_code'].isin(supported_languages)]
        except KeyError:
            questions_df_supported = questions_df

        if (existing_answers_df['model'] == model_name).sum() == len(questions_df_supported):
            logging.info(f"Model {model_name} has already processed all questions. Skipping...")
            continue

        model = LiteLLM_model(model_name)

        for i in range(0, len(questions_df_supported), batch_size):
            tasks = []
            for idx, row in questions_df_supported.iloc[i:i + batch_size].iterrows():
                # logging.info(f"Starting async process for question {idx} to {model_name} with provider {provider}")
                tasks.append(process_question(model, idx, row, answers_file_path, existing_answers_df))
            df_rows = await asyncio.gather(*tasks)

            df_rows = [df for df in df_rows if df is not None]
            if len(df_rows) > 0:
                df = pd.concat(df_rows, ignore_index=True)
                df.to_csv(answers_file_path, mode='a', header=not os.path.exists(answers_file_path), index=False)

        for idx, row in questions_df_supported.iterrows():
            logging.info(f"Starting async process for question {idx} to {model} with provider {provider}")
            await process_question(model, idx, row, answers_file_path, existing_answers_df)


# async def process_unassigned_models(models, questions_df, answers_file_path, existing_answers_df):
#     for model_name in models:
#
#         try:
#             supported_languages = LITE_LLM_METADATA_DICT[model_name]['supported_languages']
#             logging.info(f"Supported languages for {model_name}: {supported_languages}. Only running those!")
#             questions_df_supported = questions_df[questions_df['language_code'].isin(supported_languages)]
#         except KeyError:
#             questions_df_supported = questions_df
#
#         if (existing_answers_df['model'] == model_name).sum() == len(questions_df_supported):
#             logging.info(f"Model {model_name} has already processed all questions. Skipping...")
#             continue
#
#         model = LangChainModel(streaming=False, model=model_name)
#
#         for idx, row in questions_df_supported.iterrows():
#             logging.info(f"Starting async process for question {idx} to {model}")
#             await process_question(model, idx, row, answers_file_path, existing_answers_df)


def init_experiment(experiment_dir):
    if not os.path.isdir(experiment_dir):
        raise ValueError(f"Experiment directory {experiment_dir} does not exist.")
    if experiment_dir.endswith('/'):
        experiment_dir = experiment_dir[:-1]

    questions_file_path = get_questions_path(experiment_dir)
    answers_file_path = get_answers_path(experiment_dir)

    setup_logging(experiment_dir)
    logging.info("Running Unified API")
    logging.info(f"Experiment {os.path.basename(experiment_dir)}")

    existing_answers_df = load_existing_answers(answers_file_path)
    logging.info(existing_answers_df.head())
    for col in ['question_idx', 'model', 'final_response']:
        if col not in existing_answers_df.columns:
            existing_answers_df[col] = pd.NA

    # Questions are now formatted in the file already
    questions_df = pd.read_csv(questions_file_path, index_col='question_idx')
    logging.info(questions_df.head())
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    return questions_df, answers_file_path, existing_answers_df


async def run_experiment(questions_df, answers_file_path, existing_answers_df, batch_size=1, model_list=None):
    models_by_provider, unassigned_models = group_models_by_provider(model_list)
    tasks = []

    # Process other models normally
    for provider, models in models_by_provider.items():
        # We do these models with batching separately
        # if not any(m.startswith(("gpt-", "text-", "claude-")) for m in models):
        task = process_questions_for_provider(provider, models, questions_df, answers_file_path,
                                              existing_answers_df, batch_size=batch_size)
        tasks.append(task)

    await asyncio.gather(*tasks)


def run_local_models(questions_df, answers_file_path, existing_answers_df, cache_dir, batch_size=1, model_list=None):
    for model_name in model_list:
        if (existing_answers_df['model'] == model_name).sum() == len(questions_df):
            logging.info(f"Model {model_name} has already processed all questions. Skipping...")
            continue

        # loading the model
        model = LOCAL_LLMS_CLASSES[model_name](cache_dir)

        if model_name in LOCAL_LLM_METADATA_DICT.keys():
            supported_languages = LOCAL_LLM_METADATA_DICT[model_name]['supported_languages']
            logging.info(f"Supported languages for {model_name}: {supported_languages}. Only running those!")
            questions_df_supported = questions_df[questions_df['language_code'].isin(supported_languages)]
        else:
            questions_df_supported = questions_df

        # iterating over the questions
        for i in range(0, len(questions_df_supported), batch_size):
            row = questions_df_supported.iloc[i:i + batch_size]
            idx = [i for i in row.index]
            logging.info(f"Starting local model process for question {idx} to {model}")
            model.process_question(idx, row, answers_file_path, existing_answers_df)

        # deleting the model from memory
        del model
        gc.collect()
        torch.cuda.empty_cache()


def post_process(questions_df, answers_file_path):
    logging.info("Starting post-processing")
    try:
        df = pd.read_csv(answers_file_path, on_bad_lines='warn')
        logging.info("CSV file read successfully!")
    except Exception as e:
        logging.info(f"Error reading CSV file: {e}")

    idx_to_extract = ~df['final_response'].isna()
    if 'extracted' in df.columns:
        idx_to_extract = idx_to_extract & (df['extracted'].isna())

    chat_completor = ChatCompletor(os.getenv("OPENAI_API_KEY"))
    tqdm.pandas()
    messages = []
    for _, row in df[idx_to_extract].iterrows():
        answer_scale = questions_df.loc[row['question_idx'], 'answer_scale']
        answer_scale = ast.literal_eval(answer_scale)
        messages.append(generate_message(row['final_response'], answer_scale))

    df.loc[idx_to_extract, "extracted"] = asyncio.run(chat_completor.call_chatgpt_bulk(messages))

    df.to_csv(answers_file_path.replace(".csv", "_extracted.csv"), index=False)
    logging.info(f"Post-processed data saved to {answers_file_path}")


def run_all(experiment_dir, no_api=False, no_local=False, post_process_only=False, cache_dir=None, batch_size=1,
            model=None):
    questions_df, answers_file_path, existing_answers_df = init_experiment(experiment_dir)

    # running async part of the experiment
    if not post_process_only:
        if not no_api:
            logging.info("Running API models")
            model_list = MODEL_LIST if model is None else [model]
            print(f"Running API models: {model_list}")
            asyncio.run(run_experiment(questions_df, answers_file_path, existing_answers_df, batch_size=batch_size,
                                       model_list=model_list))
        if not no_local:
            logging.info("Running local models")
            logging.info(f"Cache directory: {cache_dir}")
            logging.info(f"Cuda available: {torch.cuda.is_available()}")
            for i in range(torch.cuda.device_count()):
                logging.info(f"Cuda device {i}: {torch.cuda.get_device_name(i)}")
            model_list = LOCAL_LLMS_CLASSES.keys() if model is None else [model]
            print(f"Running local models: {model_list}")
            run_local_models(questions_df, answers_file_path, existing_answers_df, cache_dir, batch_size,
                             model_list=model_list)
    # post process the answers
    post_process(questions_df, answers_file_path)


def main(args):
    config = load_config(args.config)
    base_dir = config['experiment']['base_dir']
    source_question_file = config['experiment']['source_question_file']
    experiment_dir = setup_experiment_files(base_dir, source_question_file)

    # Save config file to experiment directory
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as file:
        yaml.safe_dump(config, file)

    run_flags = vars(args)
    run_flags.pop('config')
    run_all(experiment_dir, **run_flags)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments based on a specified configuration file.")
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for processing questions.")
    parser.add_argument('--no_api', action='store_true', help="Do not run API models.")
    parser.add_argument('--no_local', action='store_true', help="Do not run local models.")
    parser.add_argument('--model', type=str, default=None, help="Run only the specified model.")
    parser.add_argument('--cache_dir', type=str, default=DEFAULT_LLM_CACHE_DIR, help="Directory to store local model cache.")
    main(parser.parse_args())
