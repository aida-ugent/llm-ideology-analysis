import argparse
import asyncio
from dotenv import load_dotenv
import os
from typing import Dict, List
import ast

import pandas as pd
from tqdm import tqdm
from utils.chat import ChatCompletor

# Load environment variables
load_dotenv()

# Get paths from environment with defaults
API_KEYS_ENV = os.getenv('API_KEYS_ENV')
load_dotenv(dotenv_path=API_KEYS_ENV)

question_file = "questions.csv"
answer_file = "answers_extracted_checked.csv"
output_file = "answers_extracted_checked_man.csv"

MAX_SUMMARY_LENGTH = 1000
MODEL = "gpt-4o"
# MODEL = "gpt-3.5-turbo"
DEBUG = False  # Only cross-check a random subset of 100 rows.
SHOW_YES = False  # Show the 'yes' responses for debugging.


def generate_message(description_test: str, description_ref: str) -> List[Dict[str, str]]:
    system = (
        f"You are a helpful assistant specialized in comparing whether a TEST Description vaguely talks about the same "
        f"person as the REFERENCE Description. You ONLY respond with 'yes', 'no', or 'refusal'."
    )

    instruction = (
        f"I am going to give you two descriptions. Please tell me if they *vaguely* refer to the same person. The "
        f"details (like years or places) do NOT need to match, just say 'yes' if both descriptions broadly talk about "
        f"the same person and 'no' if they clearly discuss completely different people. "
        f"If the TEST Description is a complete refusal to answer or simply points to a reference, respond with "
        f"'refusal'. Please ONLY respond with 'yes', 'no', or 'refusal'."
    )

    message = [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": f"{system}\n\n"
                       f"### Instruction: {instruction}\n\n"
                       f"### REFERENCE Description : {description_ref}\n\n\n"
                       # f"This was the END of Description 1!\n\n\n"
                       f"### TEST Description: {description_test}\n\n\n"
                       # f"This was the END of Description 2!\n\n\n"
                       f"### Response:"
        },
    ]
    return message


def process(results_dir: str, topic_summaries_paths: List[str], translator: ChatCompletor) -> pd.DataFrame:
    summary_df = []
    for topics_path in topic_summaries_paths:
        with open(topics_path, 'r') as file:
            summary_df.append(pd.read_csv(file, index_col=None))
    summary_df = pd.concat(summary_df)

    questions_df = pd.read_csv(os.path.join(results_dir, question_file))
    answers_df = pd.read_csv(os.path.join(results_dir, answer_file))
    df = pd.merge(answers_df, questions_df, on="question_idx", how='left', suffixes=('', '_dupe'))
    df = pd.merge(df, summary_df, left_on='topic_idx', right_on="wikidata_id", how='left', suffixes=('', '_dupe'))

    if 'summary' in df:
        df['expected_summary'] = df['summary']
    elif 'language_code' in df:
        df['expected_summary'] = df.apply(lambda r: r[f'summary-{r["language_code"]}'], axis=1)
    else:
        raise ValueError("No summary column found (or language code to index it).")

    if DEBUG:
        df = df.sample(n=100, replace=False)

    if 'stage_1_response_valid' in answers_df:
        df = df[df['stage_1_response_valid'].isna()]
    else:
        df['stage_1_response_valid'] = pd.NA

    # df = df.dropna(subset=["stage_1_response"])
    # print(f"Rows with valid responses: {len(df)}")

    if len(df) == 0:
        print("All rows have already been processed.")
        return answers_df

    tqdm.pandas()
    messages = [
        generate_message(given_summary[:MAX_SUMMARY_LENGTH], expected_summary[:MAX_SUMMARY_LENGTH])
        for given_summary, expected_summary in df[["stage_1_response", "expected_summary"]].values
    ]
    responses = asyncio.run(translator.call_chatgpt_bulk(messages))

    def clean_response(response):
        if response is None:
            return None
        response = response.strip().lower()
        manual_translate = {
            '是': 'yes',
            '否': 'no',
            '不知道': 'refusal',
        }
        if response in manual_translate:
            return manual_translate[response]

        if response not in ["yes", "no", "refusal"]:
            return None
        else:
            return response

    responses = pd.Series(responses)
    print(responses.value_counts())
    responses = responses.apply(clean_response)
    print(responses.value_counts())
    df["stage_1_response_valid"] = responses.values

    if DEBUG:
        if SHOW_YES:
            df_to_print = df
        else:
            df_to_print = df[df['stage_1_response_valid'] != 'yes']
        for idx, row in df_to_print.iterrows():
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print(f"model: {row['model']}, topic: {row['topic']}")

            print(f"stage_1_response: {row['stage_1_response'][:MAX_SUMMARY_LENGTH]}\n")

            try:
                print(f"extracted: {row['extracted']}\n")
            except KeyError:
                pass

            print(f"stage_1_response_valid: {row['stage_1_response_valid']}\n")

            print(">>>>>")
            print(f"expected_summary: {row['expected_summary'][:MAX_SUMMARY_LENGTH]}\n")

            print("\n\n\n")

    answers_df.loc[df.index, 'stage_1_response_valid'] = df['stage_1_response_valid']
    return answers_df


def main():
    parser = argparse.ArgumentParser(
        description="Process a results dir to cross-check summaries of entities to known summaries."
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Path to the results dir."
    )
    parser.add_argument(
        '--topic_summaries_paths', nargs='+', required=True,
        help="Path to the topic summaries .csv file(s). Topics (and their files) must be in the same ordering as was "
             "provided to the modular prompt expansion script."
    )  # There are no unique indices for all topics at the moment.

    args = parser.parse_args()
    results_dir = args.results_dir
    topic_summaries_paths = args.topic_summaries_paths

    chat_completor = ChatCompletor(os.getenv("OPENAI_API_KEY"), model=MODEL, chunk_size=1000)
    df = process(results_dir, topic_summaries_paths, chat_completor)

    output_path = os.path.join(results_dir, output_file)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
