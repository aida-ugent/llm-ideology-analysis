from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import torch
import logging
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Huggingface token
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN environment variable is not set")

# Parent class for all models
class Parent:
    name = None
    hf_name = None

    def __init__(self, cache_dir, load_model=True):
        self.cache_dir = cache_dir
        if load_model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_name, trust_remote_code=True,
                                                           cache_dir=self.cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(self.hf_name, torch_dtype=torch.bfloat16,
                                                              trust_remote_code=True, device_map='auto',
                                                              cache_dir=self.cache_dir).eval()
            self.input_device = next(self.model.parameters()).device
        else:
            self.input_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model_path = model_path

    def wrap_question_as_message(self, question):
        return [{"content": question, "role": "user"}]

    def get_response(self, idx, stage, messages, langcode=None):
        raise NotImplementedError

    def process_question(self, idxs, rows, answers_file_path, existing_answers_df):
        logging.debug(rows)
        previous_responses = {}
        messages = []
        for i in range(len(idxs)):
            messages.append([])
        # try:
        # Check if the current question and model combination already exists
        # Fix the logic to correctly identify if the question for the specific model has been processed

        idxs_already_processed = []
        for idx in idxs:
            question_processed = not existing_answers_df.loc[
                (existing_answers_df['question_idx'] == idx) & (existing_answers_df['model'] == self.name)
                ].empty

            if question_processed:
                logging.info(f"Question {idx} for model {self.name} has already been processed. Skipping...")
                idxs_already_processed.append(idx)
        for idx in idxs_already_processed:
            idxs.remove(idx)
        if len(idxs) == 0:
            logging.info(f"No questions to process in this batch for model {self.name}. Skipping this batch...")
            return
        rows = rows.drop(idxs_already_processed)

        result_row = {
            "question_idx": 0,
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
        result_rows = []
        for i in range(len(idxs)):
            result_rows.append(result_row.copy())

        for stage in (key for key in rows.keys() if key.startswith('stage_')):

            questions = rows[stage]
            pass_stage = False
            for i, (idx, question) in enumerate(zip(idxs, questions)):
                result_rows[i]["question_idx"] = idx
                if isinstance(question, float):
                    question = ''

                question = question.replace("<ANS>", previous_responses.get(idx, ''))

                if '<RESET>' in question:
                    messages[i] = []
                    question = question.replace("<RESET>", '')

                if question == "":
                    logging.info(
                        f"Skipping stage {stage} message for question {idx} to model {self.name}, as it is empty.")
                    pass_stage = True
                    continue

                result_rows[i][stage] = question
                messages[i].extend(self.wrap_question_as_message(question))

            if pass_stage:
                continue

            logging.info(f"Sending messages to model {self.name} for question {idxs} stage {stage}: {messages}")
            try:
                language_code = rows['language_code']
                logging.info(f"Language code: {language_code}")
            except KeyError:
                language_code = ['en' for _ in range(len(idxs))]
                logging.info(f"No language code found! Using en instead")
            response_strs, total_cost_usd = self.get_response(idxs, stage, messages, language_code)

            if response_strs:
                for i, (idx, response_str) in enumerate(zip(idxs, response_strs)):
                    previous_responses[idx] = response_str
                    result_rows[i][f"{stage}_response"] = response_str
                    messages[i].extend([{"content": response_str, "role": "assistant"}])
                    logging.info(
                        f"Got response from model {self.name} for question {idx} stage {stage}: {response_str}")
            else:
                logging.error(f"Failed to get response for model {self.name}, question {idx}, stage {stage}")

        # Explicitly store the final answer (could do sentiment analysis or other processing here)
        for i, (idx, previous_response) in enumerate(previous_responses.items()):
            result_rows[i]["final_response"] = previous_response
            result_rows[i]["total_cost_usd"] = total_cost_usd
            # After going through all stages
        df = pd.DataFrame(result_rows)
        df.to_csv(answers_file_path, mode='a', header=not os.path.exists(answers_file_path), index=False)
        # except Exception as e:
        #     logging.error(f"Unexpected error processing question {idx}: {e}")

class Jais(Parent):
    name = "jais"
    hf_name = "inceptionai/jais-family-30b-8k-chat"
    context_size = 8000

    def __init__(self, cache_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_name, 
                                                      trust_remote_code=True, 
                                                      token=HUGGINGFACE_TOKEN,
                                                      padding_side='left', 
                                                      cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_name, 
                                                         torch_dtype=torch.float16,
                                                         trust_remote_code=True, 
                                                         device_map='auto',
                                                         token=HUGGINGFACE_TOKEN, 
                                                         cache_dir=cache_dir).eval()
        self.input_device = next(self.model.parameters()).device

    # Create prompt for the model from messages
    def create_prompt(self, messages, langcode):
        if langcode == "en":
            prompt = "### Instruction:Your name is 'Jais', and you are named after Jebel Jais, the highest mountain in UAE. You were made by 'Inception' in the UAE. You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Complete the conversation between [|Human|] and [|AI|]:\n"
        else:
            prompt = "### Instruction:اسمك \"جيس\" وسميت على اسم جبل جيس اعلى جبل في الامارات. تم بنائك بواسطة Inception في الإمارات. أنت مساعد مفيد ومحترم وصادق. أجب دائمًا بأكبر قدر ممكن من المساعدة، مع الحفاظ على البقاء أمناً. أكمل المحادثة بين [|Human|] و[|AI|] :\n### Input:[|Human|] {Question}\n[|AI|]\n### Response :"

        for message in messages:
            if message["role"] == "user":
                prompt += f"### Input: [|Human|] {message['content']}\n"
            if message["role"] == "assistant":
                prompt += f"### Response: [|AI|] {message['content']}\n"
        prompt += "### Response: [|AI|]"
        return prompt

    # Get response from the model
    def get_response(self, idx, stage, messages, langcode=None):
        total_token_cost_usd = 0
        prompts = [self.create_prompt(message, l) for message, l in zip(messages, langcode)]
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")

        input_len = inputs["input_ids"].shape[-1]
        inputs = inputs.to(self.model.device)

        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=self.context_size,
            repetition_penalty=1.2,
            do_sample=True,
            min_new_tokens=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if stage == "stage_1":
            response = self.tokenizer.batch_decode(outputs[:, input_len:self.context_size - 100], skip_special_tokens=True)
        else:
            response = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
        # response = self.tokenizer.batch_decode(
        #     generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        # )[0]
        # response = response.split("### Response: [|AI|]")[-1]
        # logging.info(f"Got response from model {self.name} for question {idx} stage {stage}: {response}")

        return response, total_token_cost_usd

class Vikhr_Nemo(Parent):
    name = "Vikhr-Nemo"
    hf_name = "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24"
    context_size = 16000

    def __init__(self, cache_dir, load_model=True):
        self.cache_dir = cache_dir
        if load_model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_name, trust_remote_code=True, padding_side='left',
                                                           cache_dir=self.cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(self.hf_name, torch_dtype=torch.bfloat16,
                                                              trust_remote_code=True, device_map='auto',
                                                              cache_dir=self.cache_dir).eval()
            self.input_device = next(self.model.parameters()).device
        else:
            self.input_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get response from the model
    def get_response(self, idx, stage, messages, langcode=None):
        response = None
        total_token_cost_usd = 0
        try:
            inputs = self.tokenizer.apply_chat_template(messages,
                                                        add_generation_prompt=True,
                                                        padding=True,
                                                        return_dict=True,
                                                        return_tensors="pt",
                                                        )
            input_len = inputs["input_ids"].shape[-1]
            inputs = inputs.to(self.input_device)
            mex_new_tokens = self.context_size - input_len

            outputs = self.model.generate(**inputs,
                                          max_new_tokens=mex_new_tokens,
                                          repetition_penalty=1.2,
                                          pad_token_id=self.tokenizer.pad_token_id,
                                          eos_token_id=self.tokenizer.eos_token_id)

            if stage == "stage_1":
                response = self.tokenizer.batch_decode(outputs[:, input_len:self.context_size - 100], skip_special_tokens=True)
            else:
                response = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
            # print(response)
            # logging.info(f"Got response from model {self.name} for question {idx} stage {stage}: {response}")

        except Exception as e:
            logging.error(f"Attempt failed for question {idx}, stage {stage}, model {self.name}: {e}")

        return response, total_token_cost_usd


class Silma(Parent):
    name = "silma-9b"
    hf_name = "silma-ai/SILMA-9B-Instruct-v1.0"
    context_size = 8192

    def __init__(self, cache_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_name, trust_remote_code=True, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_name, torch_dtype=torch.float16,
                                                          trust_remote_code=True, device_map='auto',
                                                          cache_dir=cache_dir).eval()
        self.input_device = next(self.model.parameters()).device

    # Get response from the model
    def get_response(self, idx, stage, messages, langcode=None):
        response = None
        total_token_cost_usd = 0
        try:
            inputs = self.tokenizer.apply_chat_template(messages,
                                                        add_generation_prompt=True,
                                                        padding=True,
                                                        return_dict=True,
                                                        return_tensors="pt",
                                                        )
            input_len = inputs["input_ids"].shape[-1]
            inputs = inputs.to(self.input_device)
            mex_new_tokens = self.context_size - input_len

            outputs = self.model.generate(**inputs,
                                          max_new_tokens=mex_new_tokens,
                                          repetition_penalty=1.2,
                                          eos_token_id=self.tokenizer.eos_token_id)

            if stage == "stage_1":
                response = self.tokenizer.batch_decode(outputs[:, input_len:self.context_size - 100], skip_special_tokens=True)
            else:
                response = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
            # print(response)
            # logging.info(f"Got response from model {self.name} for question {idx} stage {stage}: {response}")

        except Exception as e:
            logging.error(f"Attempt failed for question {idx}, stage {stage}, model {self.name}: {e}")

        return response, total_token_cost_usd


class Bainchun(Parent):
    name = "Bainchun2"
    hf_name = "baichuan-inc/Baichuan2-13B-Chat"
    context_size = 4096

    def __init__(self, cache_dir):

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_name, revision="v2.0", cache_dir=cache_dir,
                                                       use_fast=False, padding_side='left', trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_name, revision="v2.0", cache_dir=cache_dir,
                                                          device_map="auto", torch_dtype=torch.bfloat16,
                                                          trust_remote_code=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(self.hf_name, revision="v2.0",
                                                                        cache_dir=cache_dir)
        self.input_device = next(self.model.parameters()).device

    # Create prompt for the model from messages
    def create_prompt(self, messages):
        prompt = ""
        for message in messages:
            if message["role"] == "user":
                prompt += f"user: {message['content']}\n"
            if message["role"] == "assistant":
                prompt += f"bot: {message['content']}\n"
        prompt += "bot:"
        return prompt

    # Get response from the model
    def get_response(self, idx, stage, messages, langcode=None):
        response = None
        total_token_cost_usd = 0
        try:
            prompts = [self.create_prompt(message) for message in messages]
            inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
            input_len = inputs["input_ids"].shape[-1]
            inputs = inputs.to(self.model.device)
            mex_new_tokens = self.context_size - input_len

            self.model.generation_config.max_new_tokens = mex_new_tokens
            self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

            outputs = self.model.generate(**inputs)

            if stage == "stage_1":
                response = self.tokenizer.batch_decode(outputs[:, input_len:self.context_size - 100], skip_special_tokens=True)
            else:
                response = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
            # logging.info(f"Got response from model {self.name} for question {idx} stage {stage}: {response}")

        except Exception as e:
            logging.error(f"Attempt failed for question {idx}, stage {stage}, model {self.name}: {e}")

        return response, total_token_cost_usd


class Teuken(Parent):
    name = "teuken"
    hf_name = "openGPT-X/Teuken-7B-instruct-research-v0.4"
    context_size = 4096

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_name, trust_remote_code=True, use_fast=False,
                                                       padding_side='left', cache_dir=self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_name, torch_dtype=torch.bfloat16,
                                                          trust_remote_code=True, device_map='auto',
                                                          cache_dir=self.cache_dir).eval()
        self.input_device = next(self.model.parameters()).device

    # Get response from the model
    def get_response(self, idx, stage, messages, langcode=None):
        response = None
        total_token_cost_usd = 0
        if langcode is not None:
            if langcode.unique().shape[0] > 1:
                logging.error(f"Multiple language codes detected for question {idx}: {langcode}")
                return response, total_token_cost_usd
            else:
                langcode = langcode.iloc[0]
        try:
            inputs = self.tokenizer.apply_chat_template(messages, chat_template=langcode.upper(),
                                                        add_generation_prompt=True, padding=True, return_dict=True,
                                                        return_tensors="pt")
            # inputs = [
            #     self.tokenizer.apply_chat_template(message, chat_template=l.upper(), add_generation_prompt=True,
            #                                         padding=True, return_dict=True, return_tensors="pt")
            #     for message, l in zip(messages, langcode)
            # ]
            # inputs = self.tokenizer.pad(inputs, return_tensors="pt")
            # inputs = self.tokenizer.apply_chat_template(messages,
            #                                             chat_template=langcode.upper(),
            #                                             add_generation_prompt=True,
            #                                             padding=True,
            #                                             return_dict=True,
            #                                             return_tensors="pt",
            #                                             )
            input_len = inputs["input_ids"].shape[-1]
            inputs = inputs.to(self.input_device)
            mex_new_tokens = self.context_size - input_len

            outputs = self.model.generate(input_ids=inputs["input_ids"],
                                          attention_mask=inputs["attention_mask"],
                                          max_new_tokens=mex_new_tokens,
                                          repetition_penalty=1.2,
                                          pad_token_id=self.tokenizer.pad_token_id,
                                          eos_token_id=self.tokenizer.eos_token_id,
                                          use_cache=True)  # this is 10x speedup

            if stage == "stage_1":
                response = self.tokenizer.batch_decode(outputs[:, input_len:self.context_size - 100], skip_special_tokens=True)
            else:
                response = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
            # print(response)
            # logging.info(f"Got response from model {self.name} for question {idx} stage {stage}: {response}")

        except Exception as e:
            logging.error(f"Attempt failed for question {idx}, stage {stage}, model {self.name}: {e}")

        return response, total_token_cost_usd


class EuroLLM(Parent):
    name = "euroLLM"
    hf_name = "utter-project/EuroLLM-9B-Instruct"
    context_size = 4096

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_name, 
                                                      trust_remote_code=True,
                                                      cache_dir=self.cache_dir, 
                                                      padding_side='left',
                                                      token=HUGGINGFACE_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_name, 
                                                         torch_dtype=torch.bfloat16,
                                                         trust_remote_code=True, 
                                                         device_map='auto',
                                                         cache_dir=self.cache_dir, 
                                                         token=HUGGINGFACE_TOKEN).eval()
        self.input_device = next(self.model.parameters()).device

    # Get response from the model
    def get_response(self, idx, stage, messages, langcode=None):
        response = None
        total_token_cost_usd = 0
        try:
            for i in range(len(messages)):
                messages[i].insert(0, {"role": "system",
                                       "content": "You are EuroLLM --- an AI assistant specialized in European languages that provides safe, educational and helpful answers."})

            inputs = self.tokenizer.apply_chat_template(messages,
                                                        add_generation_prompt=True,
                                                        padding=True,
                                                        return_dict=True,
                                                        return_tensors="pt",
                                                        )
            input_len = inputs["input_ids"].shape[-1]
            inputs = inputs.to(self.input_device)
            mex_new_tokens = self.context_size - input_len

            outputs = self.model.generate(input_ids=inputs["input_ids"],
                                          attention_mask=inputs["attention_mask"],
                                          max_new_tokens=mex_new_tokens,
                                          repetition_penalty=1.2,
                                          pad_token_id=self.tokenizer.pad_token_id,
                                          eos_token_id=self.tokenizer.eos_token_id)

            if stage == "stage_1":
                response = self.tokenizer.batch_decode(outputs[:, input_len:self.context_size - 100], skip_special_tokens=True)
            else:
                response = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
            # print(response)
            # logging.info(f"Got response from model {self.name} for question {idx} stage {stage}: {response}")

        except Exception as e:
            logging.error(f"Attempt failed for question {idx}, stage {stage}, model {self.name}: {e}")

        return response, total_token_cost_usd
