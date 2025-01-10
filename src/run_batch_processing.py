import os
import yaml
import logging
import argparse
from dotenv import load_dotenv
import asyncio
import pandas as pd
from typing import List, Dict, Any

# Load environment variables
load_dotenv()  # Load default .env first
api_keys_env = os.getenv('API_KEYS_ENV')
if api_keys_env:
    load_dotenv(dotenv_path=api_keys_env)

from llms import LITE_LLM_LLMS
from utils import (setup_logging, load_existing_answers, load_config, 
                  setup_experiment_files, get_questions_path, get_answers_path)
from batch_processors import AnthropicBatchProcessor, OpenAIBatchProcessor

async def process_with_batches(questions_df: pd.DataFrame,
                             answers_file_path: str,
                             existing_answers_df: pd.DataFrame,
                             model_list: List[str],
                             batch_size: int = 50000,
                             reprocess_failed: bool = False) -> None:
    """Process all supported models using batch processors."""
    tasks = []
    
    # Initialize processors with appropriate batch sizes
    anthropic_processor = AnthropicBatchProcessor(batch_size=min(batch_size, 10000))  # Anthropic's limit
    openai_processor = OpenAIBatchProcessor(batch_size=min(batch_size, 50000))  # OpenAI's limit
    
    # Add tasks for each provider's batch processing
    tasks.extend([
        anthropic_processor.process_models(
            model_list, 
            questions_df, 
            answers_file_path,
            existing_answers_df, 
            reprocess_failed
        ),
        openai_processor.process_models(
            model_list, 
            questions_df, 
            answers_file_path,
            existing_answers_df, 
            reprocess_failed
        )
    ])
    
    # Run all batch processors concurrently
    await asyncio.gather(*tasks)

def init_experiment(experiment_dir: str):
    """Initialize experiment and load necessary data."""
    if not os.path.isdir(experiment_dir):
        raise ValueError(f"Experiment directory {experiment_dir} does not exist.")
    if experiment_dir.endswith('/'):
        experiment_dir = experiment_dir[:-1]

    questions_file_path = get_questions_path(experiment_dir)
    answers_file_path = get_answers_path(experiment_dir)

    setup_logging(experiment_dir)
    logging.info("Running Batch Processing")
    logging.info(f"Experiment {os.path.basename(experiment_dir)}")

    existing_answers_df = load_existing_answers(answers_file_path)
    logging.info(f"Loaded {len(existing_answers_df)} existing answers")
    
    for col in ['question_idx', 'model', 'final_response']:
        if col not in existing_answers_df.columns:
            existing_answers_df[col] = pd.NA

    questions_df = pd.read_csv(questions_file_path, index_col='question_idx')
    logging.info(f"Loaded {len(questions_df)} questions")
    
    return questions_df, answers_file_path, existing_answers_df

def main(args):
    """Main entry point for the script."""
    config = load_config(args.config)
    base_dir = config['experiment']['base_dir']
    source_question_file = config['experiment']['source_question_file']
    experiment_dir = setup_experiment_files(base_dir, source_question_file)

    # Save config file to experiment directory
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as file:
        yaml.safe_dump(config, file)

    questions_df, answers_file_path, existing_answers_df = init_experiment(experiment_dir)
    
    # Create tmp directory for OpenAI batch files if it doesn't exist
    tmp_dir = os.path.join(experiment_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    os.chdir(tmp_dir)  # Change to tmp directory for OpenAI batch files

    try:
        # Run the batch experiment
        asyncio.run(process_with_batches(
            questions_df=questions_df,
            answers_file_path=answers_file_path,
            existing_answers_df=existing_answers_df,
            model_list=LITE_LLM_LLMS,
            batch_size=args.batch_size,
            reprocess_failed=args.reprocess_failed
        ))
    finally:
        # Change back to original directory
        os.chdir(experiment_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch processing experiments.")
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    parser.add_argument('--batch-size', type=int, default=50000,
                       help="Size of batches to process (will be automatically adjusted for each provider's limits)")
    parser.add_argument('--reprocess-failed', action='store_true',
                       help="Reprocess failed questions")
    main(parser.parse_args())