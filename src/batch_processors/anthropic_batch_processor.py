import os
import logging
import pandas as pd
import asyncio
from typing import List, Dict, Any
import anthropic
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request
from .base_batch_processor import BatchProcessorBase
import time

class AnthropicBatchProcessor(BatchProcessorBase):
    def __init__(self, batch_size: int = 10000):
        super().__init__(min(batch_size, 10000))  # Anthropic's limit
        self.client = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        
    async def process_models(self, model_list: List[str], 
                           questions_df: pd.DataFrame,
                           answers_file_path: str, 
                           existing_answers_df: pd.DataFrame,
                           reprocess_failed: bool = False):
        """Process all Anthropic models from the provided list."""
        anthropic_models = self.filter_models(model_list, "anthropic")
        
        if not anthropic_models:
            logging.info("No Anthropic models to process")
            return
            
        logging.info(f"Processing Anthropic models: {anthropic_models}")
        
        for full_model_name in anthropic_models:
            model_name = self.extract_model_name(full_model_name)
            logging.info(f"Starting batch processing for Anthropic model {model_name}")
            await self.process_model(
                model_name,
                questions_df,
                answers_file_path,
                existing_answers_df,
                reprocess_failed
            )
    def _create_stage_request(self, idx: int, row: pd.Series, model_name: str, 
                            stage: int, previous_responses: Dict[int, str]) -> Dict:
        """Create a single message request for a specific stage."""
        question = row[f'stage_{stage}']
        
        if isinstance(question, float) or not question:
            return None
            
        # Replace <ANS> with previous stage response if it exists
        if '<ANS>' in question and (stage - 1) in previous_responses:
            prev_response = previous_responses[stage - 1].get(idx, {}).get('responses', {}).get(idx, '')
            question = question.replace("<ANS>", prev_response)
            
        if '<RESET>' in question:
            question = question.replace("<RESET>", '')
        
        # Create the request as a dictionary instead of Request object
        return {
            "custom_id": f"{model_name}_{idx}_{stage}",
            "params": {
                "model": model_name,
                "max_tokens": 4096,
                "messages": [{
                    "role": "user",
                    "content": question
                }]
            }
        }

    def chunk_stage_requests(self, questions_df: pd.DataFrame, model_name: str, 
                           stage: int, previous_responses: Dict[int, str]) -> List[List[Request]]:
        """Split requests for a specific stage into chunks."""
        all_requests = []
        current_chunk = []
        
        for idx, row in questions_df.iterrows():
            request = self._create_stage_request(idx, row, model_name, stage, previous_responses)
            if request:
                current_chunk.append(request)
                
                if len(current_chunk) >= self.batch_size:
                    all_requests.append(current_chunk)
                    current_chunk = []
                    
        if current_chunk:
            all_requests.append(current_chunk)
            
        return all_requests

    def format_stage_results(self, results: List[Dict[str, Any]], stage: int) -> Dict[int, str]:
        """Format results for a specific stage."""
        stage_responses = {}
        
        for result in results:
            if result["status"] != "succeeded":
                logging.warning(f"Request {result['custom_id']} failed with status {result['status']}")
                continue
                
            model_name, idx, stage_num = result["custom_id"].split("_")
            idx = int(idx)
            
            if idx not in stage_responses:
                stage_responses[idx] = result.get("response", "")
                
        return stage_responses

    async def process_model(self, model_name: str, questions_df: pd.DataFrame, 
                          answers_file_path: str, existing_answers_df: pd.DataFrame,
                          reprocess_failed: bool = False):
        """Process all questions for a specific model using staged batching."""
        # Filter questions as before
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
            1: {'questions': {}, 'responses': {}},
            2: {'questions': {}, 'responses': {}},
            3: {'questions': {}, 'responses': {}}
        }  # Store both questions and responses for each stage
        
        for stage in range(1, 4):
            logging.info(f"Processing stage {stage} for model {model_name}")
            
            # Store the stage questions first
            for idx, row in questions_to_process.iterrows():
                question = row.get(f'stage_{stage}')
                if not pd.isna(question):
                    stage_responses[stage]['questions'][idx] = question
            
            # Create and process batches
            request_chunks = self.chunk_stage_requests(
                questions_to_process, 
                model_name, 
                stage, 
                stage_responses
            )
            
            stage_results = []
            
            # Process each batch for this stage
            for chunk_idx, chunk in enumerate(request_chunks):
                if not chunk:  # Skip if no valid requests in chunk
                    continue
                    
                logging.info(f"Processing stage {stage} chunk {chunk_idx + 1}/{len(request_chunks)}")
                results = await self.process_batch(chunk)
                stage_results.extend(results)
                
                # Small delay between chunks
                await asyncio.sleep(1)
            
            # Store responses for this stage
            stage_results = self.format_stage_results(stage_results, stage)
            stage_responses[stage]['responses'].update(stage_results)
            
            # Save intermediate results
            self.format_final_results(stage_responses, model_name, answers_file_path)

    def format_final_results(self, stage_responses: Dict[int, Dict[str, Dict[int, str]]], 
                           model_name: str, answers_file_path: str):
        """Format and save all stage results to CSV."""
        formatted_results = []
        
        # Get all unique question indices
        all_indices = set()
        for stage in stage_responses.values():
            all_indices.update(stage['questions'].keys())
            all_indices.update(stage['responses'].keys())
            
        for idx in all_indices:
            row_data = {
                "question_idx": idx,
                "model": model_name,
                "total_cost_usd": 0,  # Calculate if needed
                "stage_1": stage_responses[1]['questions'].get(idx, pd.NA),
                "stage_1_response": stage_responses[1]['responses'].get(idx, pd.NA),
                "stage_2": stage_responses[2]['questions'].get(idx, pd.NA),
                "stage_2_response": stage_responses[2]['responses'].get(idx, pd.NA),
                "stage_3": stage_responses[3]['questions'].get(idx, pd.NA),
                "stage_3_response": stage_responses[3]['responses'].get(idx, pd.NA),
                "final_response": stage_responses[3]['responses'].get(idx, pd.NA),
                "extracted": pd.NA,
            }
            formatted_results.append(row_data)
            
        if formatted_results:
            df = pd.DataFrame(formatted_results)
            df.to_csv(answers_file_path, mode='a', header=not os.path.exists(answers_file_path), index=False)

    async def process_batch(self, requests: List[Request]) -> List[Dict[str, Any]]:
        """Process a single batch of requests."""
        try:
            logging.info(f"Sending batch with {len(requests)} requests")
            
            # Create the batch
            message_batch = self.client.beta.messages.batches.create(requests=requests)
            batch_id = message_batch.id
            logging.info(f"Created batch {batch_id} with {len(requests)} requests")
            
            # Poll until processing is complete
            start_time = time.time()
            final_status = None
            
            while True:
                try:
                    batch_status = self.client.beta.messages.batches.retrieve(batch_id)
                    total_processed = (batch_status.request_counts.succeeded + 
                                     batch_status.request_counts.errored + 
                                     batch_status.request_counts.expired)
                    
                    logging.info(f"Batch {batch_id} status: {batch_status.processing_status}\n"
                               f"Succeeded: {batch_status.request_counts.succeeded}\n"
                               f"Errored: {batch_status.request_counts.errored}\n"
                               f"Expired: {batch_status.request_counts.expired}\n"
                               f"Processing: {batch_status.request_counts.processing}\n"
                               f"Time elapsed: {int(time.time() - start_time)}s")
                    
                    if batch_status.processing_status == "ended":
                        final_status = batch_status
                        break
                        
                    if time.time() - start_time > 3600:  # 1 hour timeout
                        logging.error("Batch processing timeout after 1 hour")
                        break
                        
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logging.error(f"Error checking batch status: {e}")
                    await asyncio.sleep(30)
            
            if not final_status or final_status.processing_status != "ended":
                logging.error("Batch processing did not complete successfully")
                return []
            
            # Process results
            results = []
            try:
                logging.info(f"Starting to retrieve results for batch {batch_id}")
                result_count = 0
                
                for result in self.client.beta.messages.batches.results(batch_id):
                    result_count += 1
                    response_data = {
                        "custom_id": result.custom_id,
                        "status": result.result.type
                    }
                    
                    match result.result.type:
                        case "succeeded":
                            response_data["response"] = result.result.message.content[0].text
                            response_data["usage"] = result.result.message.usage
                            logging.info(f"Processed successful result for {result.custom_id}")
                        case "errored":
                            error_msg = result.result.error
                            logging.error(f"Request {result.custom_id} failed: {error_msg}")
                            response_data["error"] = error_msg
                        case "expired":
                            logging.error(f"Request expired for {result.custom_id}")
                        case _:
                            logging.error(f"Unknown result type {result.result.type} for {result.custom_id}")
                    
                    results.append(response_data)
                
                logging.info(f"Retrieved {result_count} results for batch {batch_id}")
                return results
                
            except Exception as e:
                logging.error(f"Error processing batch results: {e}")
                logging.exception(e)  # This will print the full traceback
                return []
                
        except Exception as e:
            logging.error(f"Error in batch processing: {e}")
            logging.exception(e)  # This will print the full traceback
            return []

