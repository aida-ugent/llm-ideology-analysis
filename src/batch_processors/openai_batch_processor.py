import os
import json
import logging
import pandas as pd
import asyncio
from typing import List, Dict, Any
from openai import OpenAI
from datetime import datetime, timedelta
from .base_batch_processor import BatchProcessorBase

class OpenAIBatchProcessor(BatchProcessorBase):
    def __init__(self, batch_size: int = 50000):
        super().__init__(min(batch_size, 50000))  # OpenAI's limit
        self.client = OpenAI()
        self.COMPLETION_WINDOW = "24h"
        
    def create_stage_batch_file(self, questions_df: pd.DataFrame, model_name: str, 
                              stage: int, previous_responses: Dict[int, str]) -> str:
        """Create a JSONL file for a specific stage batch processing."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_input_stage_{stage}_{timestamp}.jsonl"
        
        with open(filename, 'w') as f:
            for idx, row in questions_df.iterrows():
                question = row[f'stage_{stage}']
                if isinstance(question, float) or not question:
                    continue
                    
                # Replace <ANS> with previous stage response
                if '<ANS>' in question and (stage - 1) in previous_responses:
                    question = question.replace("<ANS>", previous_responses[stage - 1].get(idx, ''))
                    
                messages = []
                if '<RESET>' in question:
                    question = question.replace("<RESET>", '')
                    
                messages.append({"role": "user", "content": question})
                
                request = {
                    "custom_id": f"{model_name}_{idx}_{stage}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": messages,
                        "max_tokens": 4096
                    }
                }
                f.write(json.dumps(request) + '\n')
        
        return filename

    async def process_model(self, model_name: str, questions_df: pd.DataFrame, 
                          answers_file_path: str, existing_answers_df: pd.DataFrame,
                          reprocess_failed: bool = False):
        """Process all questions for a specific model using staged OpenAI batching."""
        # Filter questions
        model_answers = existing_answers_df[existing_answers_df['model'] == model_name]
        
        if reprocess_failed:
            processed_indices = model_answers[
                model_answers['final_response'].notna() & 
                (model_answers['final_response'] != '')
            ]['question_idx'].tolist()
        else:
            processed_indices = model_answers['question_idx'].tolist()
            
        questions_to_process = questions_df.drop(processed_indices, errors='ignore')
        
        if questions_to_process.empty:
            logging.info(f"No questions to process for model {model_name}")
            return

        # Process each stage sequentially
        stage_responses = {
            1: {},  # Will store both questions and responses for stage 1
            2: {},  # Will store both questions and responses for stage 2
            3: {}   # Will store both questions and responses for stage 3
        }
        
        for stage in range(1, 4):
            logging.info(f"Processing stage {stage} for model {model_name}")
            
            # Store the stage questions
            for idx, row in questions_to_process.iterrows():
                question = row.get(f'stage_{stage}')
                if not pd.isna(question):
                    if stage not in stage_responses:
                        stage_responses[stage] = {}
                    stage_responses[stage][idx] = question
            
            # Create batch file for this stage
            input_filename = self.create_stage_batch_file(
                questions_to_process, 
                model_name, 
                stage, 
                stage_responses
            )
            
            try:
                # Process batch
                batch = await self.process_batch(input_filename)
                if not batch:
                    logging.error(f"Failed to create batch for model {model_name} stage {stage}")
                    continue
                    
                # Wait for completion
                final_status = await self.wait_for_batch_completion(batch.id)
                
                if final_status.status == "completed":
                    # Process results for this stage
                    results = self.process_batch_results(final_status.output_file_id)
                    if results:
                        stage_responses[stage].update(self.format_stage_results(results, stage))
                        logging.info(f"Stage {stage} completed with {len(results)} responses")
                        
                        # Save intermediate results after each stage
                        self.format_final_results(stage_responses, model_name, answers_file_path)
                    else:
                        logging.error(f"No results for stage {stage}")
                else:
                    logging.error(f"Batch {batch.id} failed with status {final_status.status}")
                    
            except Exception as e:
                logging.error(f"Error processing stage {stage}: {e}")
                
            finally:
                # Cleanup
                if os.path.exists(input_filename):
                    os.remove(input_filename)

    def format_stage_results(self, results: List[Dict[str, Any]], stage: int) -> Dict[int, str]:
        """Format results for a specific stage."""
        stage_responses = {}
        
        for result in results:
            try:
                if result.get("status") != "succeeded":
                    logging.warning(f"Request {result.get('custom_id')} failed with status {result.get('status')}")
                    continue
                    
                model_name, idx, stage_num = result["custom_id"].split("_")
                idx = int(idx)
                
                # Get the response text from the OpenAI response format
                response_text = result.get("response", "")
                logging.info(f"Processing response for idx {idx}, stage {stage_num}: {response_text[:100]}...")
                
                stage_responses[idx] = response_text
                
            except Exception as e:
                logging.error(f"Error processing result {result}: {e}")
                
        logging.info(f"Processed {len(stage_responses)} responses for stage {stage}")
        return stage_responses

    def format_final_results(self, stage_responses: Dict[int, Dict[int, str]], 
                           model_name: str, answers_file_path: str):
        """Format and save all stage results to CSV."""
        formatted_results = []
        
        # Get all unique question indices across all stages
        all_indices = set()
        for stage_data in stage_responses.values():
            all_indices.update(stage_data.keys())
            
        for idx in all_indices:
            row_data = {
                "question_idx": idx,
                "model": model_name,
                "total_cost_usd": 0,  # Calculate if needed
                "stage_1": stage_responses.get(1, {}).get(idx, pd.NA),  # Store both questions and responses
                "stage_1_response": stage_responses.get(1, {}).get(idx, pd.NA),
                "stage_2": stage_responses.get(2, {}).get(idx, pd.NA),
                "stage_2_response": stage_responses.get(2, {}).get(idx, pd.NA),
                "stage_3": stage_responses.get(3, {}).get(idx, pd.NA),
                "stage_3_response": stage_responses.get(3, {}).get(idx, pd.NA),
                "final_response": stage_responses.get(3, {}).get(idx, pd.NA),
                "extracted": pd.NA,
            }
            formatted_results.append(row_data)
            
        if formatted_results:
            df = pd.DataFrame(formatted_results)
            df.to_csv(answers_file_path, mode='a', header=not os.path.exists(answers_file_path), index=False)
            logging.info(f"Saved {len(formatted_results)} results to {answers_file_path}")

    async def process_batch(self, input_filename: str) -> Dict[str, Any]:
        """Upload file and process batch, returning batch details."""
        try:
            # Upload the file
            batch_file = self.client.files.create(
                file=open(input_filename, "rb"),
                purpose="batch"
            )
            
            # Create the batch
            batch = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window=self.COMPLETION_WINDOW
            )
            
            return batch
            
        except Exception as e:
            logging.error(f"Error creating batch: {e}")
            return None

    async def wait_for_batch_completion(self, batch_id: str) -> Dict[str, Any]:
        """Poll until batch is complete."""
        while True:
            batch_status = self.client.batches.retrieve(batch_id)
            
            if batch_status.status in ["completed", "failed", "expired", "cancelled"]:
                return batch_status
                
            await asyncio.sleep(30)  # Check every 30 seconds

    def process_batch_results(self, output_file_id: str) -> List[Dict[str, Any]]:
        """Process the results from a completed batch."""
        try:
            file_response = self.client.files.content(output_file_id)
            results = []
            
            for line in file_response.text.strip().split('\n'):
                result = json.loads(line)
                results.append({
                    "custom_id": result["custom_id"],
                    "status": "succeeded" if result["response"]["status_code"] == 200 else "failed",
                    "response": result["response"]["body"]["choices"][0]["message"]["content"] 
                        if result["response"]["status_code"] == 200 else None,
                    "error": result["error"] if "error" in result else None,
                    "usage": result["response"]["body"]["usage"] 
                        if result["response"]["status_code"] == 200 else None
                })
                
            return results
            
        except Exception as e:
            logging.error(f"Error processing batch results: {e}")
            return []

    def format_results(self, results: List[Dict[str, Any]], answers_file_path: str):
        """Format and save batch results to CSV."""
        formatted_results = []
        
        for result in results:
            model_name, idx = result["custom_id"].split("_")
            idx = int(idx)
            
            row_data = {
                "question_idx": idx,
                "model": model_name,
                "total_cost_usd": 0,  # Calculate based on usage if needed
                "stage_1": pd.NA,
                "stage_1_response": pd.NA,
                "stage_2": pd.NA,
                "stage_2_response": pd.NA,
                "stage_3": pd.NA,
                "stage_3_response": pd.NA,
                "final_response": result.get("response", pd.NA),
                "extracted": pd.NA,
            }
            
            if result["status"] != "succeeded":
                logging.warning(f"Request {result['custom_id']} failed with status {result['status']}")
                continue
                
            formatted_results.append(row_data)
            
        if formatted_results:
            df = pd.DataFrame(formatted_results)
            df.to_csv(answers_file_path, mode='a', header=not os.path.exists(answers_file_path), index=False)

    async def process_models(self, model_list: List[str], 
                            questions_df: pd.DataFrame,
                            answers_file_path: str, 
                            existing_answers_df: pd.DataFrame,
                            reprocess_failed: bool = False):
        """Process all OpenAI models from the provided list."""
        openai_models = self.filter_models(model_list, "openai")
        
        if not openai_models:
            logging.info("No OpenAI models to process")
            return
        
        logging.info(f"Processing OpenAI models: {openai_models}")
        
        for full_model_name in openai_models:
            model_name = self.extract_model_name(full_model_name)
            logging.info(f"Starting batch processing for OpenAI model {model_name}")
            await self.process_model(
                model_name,
                questions_df,
                answers_file_path,
                existing_answers_df,
                reprocess_failed
            )

