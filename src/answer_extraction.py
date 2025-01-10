import argparse
import asyncio
from typing import Dict, List
import ast

import pandas as pd
from tqdm import tqdm
from utils.chat import ChatCompletor


MAX_AMOUNT = None


def generate_message(input_text: str, answer_scale: List[str]) -> List[Dict[str, str]]:
    #answer_scale = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
    scale_string = ', '.join(f"'{w}'" for w in answer_scale[:-1])
    scale_string += f", and '{answer_scale[-1]}'"
    system = (
        f"You are a helpful assistant specialized in extracting the label of a message:"
        f" the possible labels are {scale_string}. If none of the labels apply reply with 'unknown'."
    )

    scale_string = ', '.join(answer_scale + ["unknown"])
    instruction = f"Determine the label of the message. Options: {scale_string}\nno other options may be given"

    message = [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": f"{system}\n\n### Instruction: {instruction}\n\n### Input: {input_text}\n\n### Response:",
        },
    ]
    return message


def process_dataframe(csv_path: str, translator: ChatCompletor) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    tqdm.pandas()
    idx_to_extract = ~df['final_response'].isna()
    if 'extracted' in df.columns:
        idx_to_extract = idx_to_extract & (df['extracted'].isna())

    print(f"Extracting labels for {idx_to_extract.sum()} messages.")

    if MAX_AMOUNT is not None:
        # Only keep first MAX_AMOUNT idx as True
        idx_to_extract = idx_to_extract & (idx_to_extract.cumsum() <= MAX_AMOUNT)

    messages = [
        generate_message(resp, ast.literal_eval(scale))
        for resp, scale in df.loc[idx_to_extract, ["final_response", "answer_scale"]].values
    ]

    translations = asyncio.run(translator.call_chatgpt_bulk(messages))

    df.loc[idx_to_extract, "extracted"] = translations

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Process a CSV file to extract message labels."
    )
    parser.add_argument(
        "--apikey", type=str, required=True, help="API Key to access the OpenAI GPT-3."
    )
    parser.add_argument(
        "--model", 
        type=str,
        help="OpenAI model to use."
    )
    parser.add_argument(
        "--max_tokens", 
        type=int,
        help="Maximum tokens for the response."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        help="Maximum number of samples per chunk."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature setting for response generation."
    )
    parser.add_argument(
        "--csv_path", type=str, required=True, help="Path to the input CSV file"
    )
    args = vars(parser.parse_args())
    csv_path = args.pop("csv_path")
    translator = ChatCompletor(**args)

    df = process_dataframe(csv_path, translator)

    df.to_csv(csv_path, index=False)
    print(f"Processed data saved to {csv_path}")


if __name__ == "__main__":
    main()
