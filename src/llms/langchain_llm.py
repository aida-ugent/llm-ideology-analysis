import logging
import asyncio
import pandas as pd
import os
import aiohttp

from langchain_community.chat_models import QianfanChatEndpoint, ChatYandexGPT, GigaChat
from langchain_core.language_models.chat_models import HumanMessage, AIMessage
# from langchain_gigachat import GigaChat

# This class will be used to interact with the models through LangChain
class LangChainModel:
    def __init__(self, streaming=True,  model="YandexGPT",):
        self.model = model
        self.name = model
                
        # Choose the appropriate chat model based on the model name
        if "yandex" in model.lower():
            self.chat = ChatYandexGPT(streaming=False, api_key = 'AQVNxLJI4tYhZuG6TQCB3_wqs3euspO81DX3YH-v',model_uri='gpt://b1gp4kikufjfvnpcq4ek/yandexgpt/latest')
        elif "giga" in model.lower():
            self.chat = GigaChat(verify_ssl_certs=False, credentials = "ODQ0MzE5YzAtZmQ2Ni00MDc0LWIwZDItZGZkYjQ2ZGJjZjE4Ojc5MGI0ZWU0LWZmMjAtNDRhOC05YTliLTE1OWM3OTBhNmIzOA==", scope="GIGACHAT_API_CORP", model = 'GigaChat-Max-preview', base_url = "https://gigachat-preview.devices.sberbank.ru/api/v1")
        else:  # Default to Qianfan/ERNIE-Bot
            self.chat = QianfanChatEndpoint(streaming=False, model="ERNIE-4.0-Turbo-8K-Latest", api_key='OCgtWARI5QgsklbNBkz3x9Sn', secret_key = 'ACuzHOnfrYeoN8IN0lE7rYbu6oRHv06G')

    async def get_response(self, messages, max_retries=3, retry_delay=10, timeout=15):
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(self.chat.ainvoke(messages), timeout=timeout)
                return response            
            except asyncio.TimeoutError:
                logging.error(f"Request timed out after {timeout} seconds. Attempt {attempt+1} of {max_retries}. Retrying...")
            except Exception as e:
                logging.error(f"An error occurred: {e}. Attempt {attempt+1} of {max_retries}. Retrying...")

            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)  # Wait before retrying
                
        logging.error(f"All {max_retries} attempts failed. Returning None.")
        return None
    

    def wrap_question_as_message(self, question):
        return HumanMessage(content=question)

    async def process_question(self, idx, row, answers_file_path, existing_answers_df):
        logging.debug(row)
        previous_response = ""
        messages = []

        question_processed = not existing_answers_df.loc[
            (existing_answers_df['question_idx'] == idx) & (existing_answers_df['model'] == self.model)
            ].empty

        if question_processed:
            logging.info(f"Question {idx} for model {self.chat.__class__.__name__} has already been processed. Skipping...")
            return

        result_row = {
            "question_idx": idx,
            "model": self.model,
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
                logging.info(f"Skipping stage {stage} message for question {idx} to model {self.chat.__class__.__name__}, as it is empty.")
                continue

            result_row[stage] = question
            message = self.wrap_question_as_message(question)
            messages.append(message)

            logging.info(f"Sending message to model for question {idx} stage {stage}: {message.content}")
            response = await self.get_response(messages)

            if response:
                response_str = response.content
                previous_response = response_str
                result_row[f"{stage}_response"] = response_str
                messages.append(AIMessage(content=response_str))
                logging.info(f"Got response from model for question {idx} stage {stage}: {response_str}")
            else:
                logging.error(f"Failed to get response for model, question {idx}, stage {stage}")

        result_row["final_response"] = previous_response
        # After going through all stages
        df = pd.DataFrame([result_row])
        df.to_csv(answers_file_path, mode='a', header=not os.path.exists(answers_file_path), index=False)

