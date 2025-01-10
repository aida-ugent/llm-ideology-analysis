from litellm import acompletion, token_counter, register_model, completion_cost
import litellm
from llms import LITE_LLM_LLMS, LITE_LLM_METADATA_DICT
import logging
import asyncio
import pandas as pd
import os

MODEL_METADATA_DICT = LITE_LLM_METADATA_DICT
register_model(MODEL_METADATA_DICT)

litellm.suppress_debug_info = True
litellm.set_verbose = False


# Brought the functions from main file to this file
# This is the class that will be used to interact with the models Through Litellm
class LiteLLM_model:
    def __init__(self, model):
        self.name = model

    async def get_response(self, idx, stage, messages, max_retries=3):
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        attempts = 0
        delay = 2  # Initial delay in seconds
        response = None
        total_token_cost_usd = 0
        while attempts < max_retries:
            try:
                prompt_token_count = token_counter(model=self.name, messages=messages)
                logging.info(f"Token count for question {idx} stage {stage} with model {self.name}: {prompt_token_count}")
                input_cost_per_token = MODEL_METADATA_DICT[self.name]['input_cost_per_token']
                output_cost_per_token = MODEL_METADATA_DICT[self.name]['output_cost_per_token']
                if "gemini" in self.name:
                    safety_settings=[
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_CIVIC_INTEGRITY",
                            "threshold": "BLOCK_NONE",
                        }
                    ]                
                    response = await acompletion(
                        model=self.name, 
                        messages=messages, 
                        input_cost_per_token=input_cost_per_token,
                        output_cost_per_token=output_cost_per_token,
                        safety_settings=safety_settings,
                    )
                else:
                    response = await acompletion(
                        model=self.name, 
                        messages=messages, 
                        input_cost_per_token=input_cost_per_token,
                        output_cost_per_token=output_cost_per_token,
                    )
                # completion_token_count = token_counter(model=model, messages=messages, )
                total_token_cost_usd = completion_cost(completion_response=response)
                return response, total_token_cost_usd

            except Exception as e:
                attempts += 1
                logging.error(f"Attempt {attempts} failed for question {idx}, stage {stage}, model {self.name}: {e}", exc_info=True)
                if attempts < max_retries:
                    await asyncio.sleep(delay)  # Wait before retrying
                    delay *= 2  # Exponential backoff
                else:
                    logging.error(f"All retry attempts failed for question {idx}, stage {stage}.")
                if response:
                    return response, total_token_cost_usd

        return response, total_token_cost_usd

    def wrap_question_as_message(self, question):
        return [{"content": question, "role": "user"}]

    async def process_question(self, idx, row, answers_file_path, existing_answers_df):
        logging.debug(row)
        previous_response = ""
        messages = []
        # try:
        # Check if the current question and model combination already exists
        # Fix the logic to correctly identify if the question for the specific model has been processed

        question_processed = not existing_answers_df.loc[
            (existing_answers_df['question_idx'] == idx) & (existing_answers_df['model'] == self.name)
            ].empty

        if question_processed:
            logging.info(f"Question {idx} for model {self.name} has already been processed. Skipping...")
            return

        result_row = {
            "question_idx": idx,
            "model": self.name,
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

            question = question.replace("<ANS>", previous_response)

            if '<RESET>' in question:
                messages = []
                question = question.replace("<RESET>", '')

            if question == "":
                logging.info(f"Skipping stage {stage} message for question {idx} to model {self.name}, as it is empty.")
                continue

            result_row[stage] = question
            messages.extend(self.wrap_question_as_message(question))

            logging.info(f"Sending messages to model {self.name} for question {idx} stage {stage}: {messages}")
            response, total_cost_usd = await self.get_response(idx, stage, messages)

            if response:
                try:
                    response_str = response['choices'][0]['message']['content']
                # The structure of gpt3.5-turbo-instruct responses is different apparently
                except AttributeError as ae:
                    response_str = response["choices"][0]["text"]
                previous_response = response_str
                result_row[f"{stage}_response"] = response_str
                messages.extend([{"content": response_str, "role": "assistant"}])
                logging.info(f"Got response from model {self.name} for question {idx} stage {stage}: {response}")
            else:
                logging.error(f"Failed to get response for model {self.name}, question {idx}, stage {stage}")

        # Explicitly store the final answer (could do sentiment analysis or other processing here)
        result_row["final_response"] = previous_response
        result_row["total_cost_usd"] = total_cost_usd
        # After going through all stages
        df = pd.DataFrame([result_row])
        df.to_csv(answers_file_path, mode='a', header=not os.path.exists(answers_file_path), index=False)
        # except Exception as e:
        #     logging.error(f"Unexpected error processing question {idx}: {e}")