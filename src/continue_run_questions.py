import argparse
import logging


from run_questions_through_unified_api import run_all
from utils import get_answers_path, load_existing_answers, setup_logging, DEFAULT_LLM_CACHE_DIR


def clean_failed_results(experiment_dir):
    setup_logging(experiment_dir)
    logging.info(f"Attempting to continue the run in {experiment_dir}")

    answers_file_path = get_answers_path(experiment_dir)
    existing_answers_df = load_existing_answers(answers_file_path)
    if len(existing_answers_df) > 0:
        logging.info(f"Loaded {len(existing_answers_df)} existing answers from {answers_file_path}")
        nan_responses = (existing_answers_df['final_response'].isna()
                         | (existing_answers_df['final_response'] == '')
                         | (existing_answers_df['stage_1_response'].isna())
                         | (existing_answers_df['stage_1_response'] == ''))
        if len(nan_responses) > 0:
            logging.info(f"Removing {nan_responses.sum()} rows with missing first or final responses")
            existing_answers_df.drop(existing_answers_df[nan_responses].index, inplace=True)
    existing_answers_df.to_csv(answers_file_path, mode='w', index=False)


def main(args):
    clean_failed_results(args.experiment_dir)
    run_all(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue the experiment in a specified results folder.")
    parser.add_argument('experiment_dir', type=str, help="Path to the experiment's results folder.")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for processing questions.")
    parser.add_argument('--no_api', action='store_true', help="Do not run API models.")
    parser.add_argument('--no_local', action='store_true', help="Do not run local models.")
    parser.add_argument('--post_process_only', action='store_true', help="Only run post_process")
    parser.add_argument('--model', type=str, default=None, help="Run only the specified model.")
    parser.add_argument('--cache_dir', type=str, default=DEFAULT_LLM_CACHE_DIR, help="Directory to store local model cache.")
    main(parser.parse_args())
