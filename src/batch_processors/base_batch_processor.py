from typing import List


class BatchProcessorBase:
    """Base class for batch processors with common utilities."""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        
    @staticmethod
    def extract_model_name(full_model_name: str) -> str:
        """Extract the actual model name from the provider/model format."""
        return full_model_name.split('/', 1)[1] if '/' in full_model_name else full_model_name
        
    @staticmethod
    def filter_models(model_list: List[str], provider_prefix: str) -> List[str]:
        """Filter models by provider prefix."""
        return [
            model for model in model_list 
            if model.startswith(f"{provider_prefix}/")
        ]

