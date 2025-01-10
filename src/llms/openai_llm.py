import os
import logging
import asyncio
import pandas as pd
from openai import OpenAI
from litellm import token_counter, completion_cost

from .llms import OPENAI_LLMS_METADATA_DICT

class OpenAIModel:
    """Base class for models using OpenAI-style APIs"""
    def __init__(self, model_name, api_key, base_url=None):
        self.name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
    async def _make_openai_request(self, messages, model_name=None):
        """Make request using OpenAI-style API and convert to litellm format"""
        if model_name is None:
            model_name = self.name.split('/')[-1]
            
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": msg["role"], "content": msg["content"]} 
                    for msg in messages
                ]
            )
        )
        
        # Convert to litellm format
        return {
            "choices": [{
                "message": {
                    "content": response.choices[0].message.content,
                    "role": "assistant"
                }
            }],
            "usage": {
                "total_tokens": 0,  # Add if token counts available
                "prompt_tokens": 0,
                "completion_tokens": 0
            }
        }

    async def get_response(self, idx, stage, messages, max_retries=3):
        attempts = 0
        delay = 2
        response = None
        total_token_cost_usd = 0
        
        while attempts < max_retries:
            try:
                prompt_token_count = token_counter(model=self.name, messages=messages)
                logging.info(f"Token count for question {idx} stage {stage} with model {self.name}: {prompt_token_count}")
                input_cost_per_token = OPENAI_LLMS_METADATA_DICT[self.name]['input_cost_per_token']
                output_cost_per_token = OPENAI_LLMS_METADATA_DICT[self.name]['output_cost_per_token']

                response = await self._make_openai_request(messages)
                total_token_cost_usd = prompt_token_count*input_cost_per_token #completion_cost(completion_response=response)
                return response, total_token_cost_usd

            except Exception as e:
                attempts += 1
                logging.error(f"Attempt {attempts} failed for question {idx}, stage {stage}, model {self.name}: {e}", exc_info=True)
                if attempts < max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    logging.error(f"All retry attempts failed for question {idx}, stage {stage}.")
                if response:
                    return response, total_token_cost_usd

        return response, total_token_cost_usd

    def wrap_question_as_message(self, question):
        """Make compatible with existing message wrapping"""
        return [{"content": question, "role": "user"}]

    async def process_question(self, idx, row, answers_file_path, existing_answers_df):
        logging.debug(row)
        previous_response = ""
        messages = []

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
                response_str = response['choices'][0]['message']['content']
                previous_response = response_str
                result_row[f"{stage}_response"] = response_str
                messages.extend([{"content": response_str, "role": "assistant"}])
                logging.info(f"Got response from model {self.name} for question {idx} stage {stage}: {response}")
            else:
                logging.error(f"Failed to get response for model {self.name}, question {idx}, stage {stage}")

        result_row["final_response"] = previous_response
        result_row["total_cost_usd"] = total_cost_usd
        
        df = pd.DataFrame([result_row])
        df.to_csv(answers_file_path, mode='a', header=not os.path.exists(answers_file_path), index=False)