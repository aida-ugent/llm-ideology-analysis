from .local_models import Jais, Vikhr_Nemo, Silma, Bainchun, Teuken, EuroLLM

LITE_LLM_LLMS = [
    # OPENAI
    #"openai/gpt-3.5-turbo",
    #"openai/gpt-3.5-turbo-instruct",
    #"openai/gpt-4-1106-preview",
    #"openai/gpt-4-0125-preview",
    # "openai/gpt-4o",
    # MISTRAL
    # "mistral/open-mixtral-8x22b",
    # "mistral/mistral-small-latest",
    #"mistral/mistral-medium-latest",
    # "mistral/mistral-large-latest",
    # ANTHROPIC
    #"anthropic/claude-3-haiku-20240307",
    #"anthropic/claude-3-5-sonnet-20240620",
    #"anthropic/claude-3-opus-20240229",
    # Claude Sonnet 3.5.1
    #"anthropic/claude-3-5-haiku-20241022",
    "anthropic/claude-3-5-sonnet-20241022",
    # TOGETHERAI
    #"together_ai/tiiuae/falcon-7b-instruct",
    #"together_ai/tiiuae/falcon-40b-instruct",
    #"together_ai/Qwen/Qwen1.5-72B-Chat",
    #"together_ai/Qwen/Qwen1.5-14B-Chat",
    "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
    #"together_ai/Qwen/Qwen2.5-14B-Instruct-Turbo",meta-llama/
    # "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
    # "together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    # "together_ai/meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    #"together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
    #"together_ai/meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    # DeepSeek
    # "deepseek/deepseek-chat",
    # DEEPINFRA
    #"deepinfra/mistralai/Mixtral-8x7B-Instruct-v0.1",
    #"deepinfra/meta-llama/Llama-2-70b-chat-hf",
    # PERPLEXITYAI
    #"perplexity/llama-3-sonar-large-32k-chat",
    #"perplexity/llama-3-sonar-large-32k-online",
    #"perplexity/llama-3-70b-instruct",
    # Deprecated from May 14
    #"perplexity/sonar-medium-chat",
    # Google AI Studio
    "gemini/gemini-pro",
    # xAI
    "xai/grok-beta",
    # "ai21/jamba-1.5-large"
]

LANGCHAIN_LLMS = [
    # Baidu AI Cloud Qianfan Platform
    #"ERNIE-Bot",  
    # Yandex Cloud
    # https://github.com/yandex-datasphere/yandex-chain
    #"YandexGPT", 
]

OPENAI_LLMS = [
    # xAI
    #"xai/grok-beta",
]

# name of the local models are the keys
# value is the class of the local model
LOCAL_LLMS_CLASSES = {
    "jais": Jais,
    # "jais_no_system_prompt": JaisNoSystemPrompt,
    "Vikhr-Nemo": Vikhr_Nemo,
    "silma": Silma,
    "bainchun" : Bainchun,
    "teuken": Teuken,
    "euro-llm": EuroLLM,
    # "sea-lion": Sea_lion,  # Results were very strange
}


LITE_LLM_METADATA_DICT = {
    "openai/gpt-3.5-turbo": {
        "max_tokens": 4097,
        "max_input_tokens": 16385,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
        "litellm_provider": "openai",
        "mode": "chat",
        "supported_languages_source": "https://help.openai.com/en/articles/8357869-how-to-change-your-language-setting-in-chatgpt#h_513834920e",
        "supported_languages": ['ar', 'zh', 'en', 'fr', 'ru', 'es']
        #"supports_function_calling": true
    },
    "openai/gpt-4-1106-preview": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00003,
        "litellm_provider": "openai",
        "mode": "chat",
        "supported_languages_source": "https://help.openai.com/en/articles/8357869-how-to-change-your-language-setting-in-chatgpt#h_513834920e",
        "supported_languages": ['ar', 'zh', 'en', 'fr', 'ru', 'es']
        #"supports_function_calling": true,
        #"supports_parallel_function_calling": true
    },
    "openai/gpt-4-0125-preview": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00003,
        "litellm_provider": "openai",
        "mode": "chat",
        #"supports_function_calling": true,
        #"supports_parallel_function_calling": true
    },
    "openai/gpt-4o": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000005,
        "output_cost_per_token": 0.0000015,
        "litellm_provider": "openai",
        "mode": "chat",
        "supported_languages_source": "https://help.openai.com/en/articles/8357869-how-to-change-your-language-setting-in-chatgpt#h_513834920e",
        "supported_languages": ['ar', 'zh', 'en', 'fr', 'ru', 'es']
        #"supports_function_calling": true,
        #"supports_parallel_function_calling": true
    },
    "mistral/mistral-small-latest": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "mistral",
        "mode": "chat",
        "supported_languages_source": "https://mistral.ai/news/mistral-large-2407/",
        "supported_languages": ['ar', 'zh', 'en', 'fr', 'ru', 'es']  
        #"supports_function_calling": true
    },
    "mistral/mistral-medium-latest": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "mistral",
        "mode": "chat",
        "supported_languages_source": "https://mistral.ai/news/mistral-large-2407/",
        "supported_languages": ['ar', 'zh', 'en', 'fr', 'ru', 'es']  
        #"supports_function_calling": true
    },
    "mistral/mistral-large-latest": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "mistral",
        "mode": "chat",
        "supported_languages_source": "https://mistral.ai/news/mistral-large-2407/",
        "supported_languages": ['ar', 'zh', 'en', 'fr', 'ru', 'es']  
        #"supports_function_calling": true
    },
    "mistral/mistral-large-2407": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000002,
        "output_cost_per_token": 0.000006,
        "litellm_provider": "mistral",
        "mode": "chat",
        "supported_languages_source": "https://mistral.ai/news/mistral-large-2407/",
        "supported_languages": ['ar', 'zh', 'en', 'fr', 'ru', 'es']  
        #"supports_function_calling": true
    },
    "mistral/open-mixtral-8x22b": {
        "max_tokens": 8191,
        "max_input_tokens": 64000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000002,
        "output_cost_per_token": 0.000006,
        "litellm_provider": "mistral",
        "mode": "chat",
        "supported_languages_source": "https://docs.mistral.ai/getting-started/models/benchmark/",
        "supported_languages": ['en', 'fr', 'es']            
        #"supports_function_calling": true
    },
    "anthropic/claude-3-haiku-20240307": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000025,
        "output_cost_per_token": 0.00000125,
        "litellm_provider": "anthropic",
        "mode": "chat",
        "supported_languages_source": "https://assets.anthropic.com/m/61e7d27f8c8f5919/original/Claude-3-Model-Card.pdf",
        "supported_languages": ['ar', 'zh', 'en', 'fr', 'ru', 'es']     
    },
    "anthropic/claude-3-opus-20240229": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075,
        "litellm_provider": "anthropic",
        "mode": "chat",
        "supported_languages_source": "https://assets.anthropic.com/m/61e7d27f8c8f5919/original/Claude-3-Model-Card.pdf",
        "supported_languages": ['ar', 'zh', 'en', 'fr', 'ru', 'es']  
    },
    "anthropic/claude-3-5-sonnet-20241022": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "anthropic",
        "mode": "chat",
        "supported_languages_source": "https://assets.anthropic.com/m/61e7d27f8c8f5919/original/Claude-3-Model-Card.pdf",
        "supported_languages": ['ar', 'zh', 'en', 'fr', 'ru', 'es']  
    },
    "together_ai/Qwen/Qwen1.5-72B-Chat": 
    {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000090,
        "output_cost_per_token": 0.00000090,
        "litellm_provider": "together_ai",
        "mode": "chat",
        "supported_languages_source": "https://qwenlm.github.io/blog/qwen1.5/",
        "supported_languages": ['ar', 'zh', 'en', 'fr', 'ru', 'es']  
    },
    "together_ai/Qwen/Qwen1.5-14B-Chat": 
    {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000030,
        "output_cost_per_token": 0.00000030,
        "litellm_provider": "together_ai",
        "mode": "chat",
        "supported_languages_source": "https://qwenlm.github.io/blog/qwen1.5/",
        "supported_languages": ['ar', 'zh', 'en', 'fr', 'ru', 'es']  
    }, 
    "deepinfra/mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "max_tokens": 8191,
        "max_input_tokens": 32768,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.00000027,
        "output_cost_per_token": 0.00000027,
        "litellm_provider": "deepinfra",
        "mode": "chat",
        "supported_languages_source": "https://docs.mistral.ai/getting-started/models/benchmark/",
        "supported_languages": ['en', 'fr', 'es']    
    },
    "deepinfra/meta-llama/Llama-2-70b-chat-hf": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000070,
        "output_cost_per_token": 0.00000090,
        "litellm_provider": "deepinfra",
        "mode": "chat",
        "supported_languages_source": "https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/",
        "supported_languages": ['en']    
    },
    "perplexity/sonar-medium-chat": {
        "max_tokens": 16384,
        "max_input_tokens": 16384,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.0000006,
        "output_cost_per_token": 0.0000018,
        "litellm_provider": "perplexity",
        "mode": "chat",
        "supported_languages_source": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md",
        "supported_languages": ['en', 'fr', 'es']    
    },
    "perplexity/llama-3-sonar-large-32k-chat": {
        "max_tokens": 32768,
        "max_input_tokens": 32768,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000001,
        "litellm_provider": "perplexity",
        "mode": "chat",
        "supported_languages_source": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md",
        "supported_languages": ['en', 'fr', 'es']    
    },
    "perplexity/llama-3-sonar-large-32k-online": {
        "max_tokens": 28000,
        "max_input_tokens": 28000,
        "max_output_tokens": 28000,
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000001,
        "litellm_provider": "perplexity",
        "mode": "online",
        "supported_languages_source": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md",
        "supported_languages": ['en', 'fr', 'es']    
    },
    "perplexity/llama-3-70b-instruct": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000001,
        "litellm_provider": "perplexity",
        "mode": "chat",
        "supported_languages_source": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md",
        "supported_languages": ['en', 'fr', 'es']    
    },
    "gemini/gemini-pro": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000015,
        "litellm_provider": "gemini",
        "mode": "chat",
        "supported_languages_source": "https://cloud.google.com/gemini/docs/codeassist/supported-languages",
        "supported_languages": ['ar', 'zh', 'en', 'fr', 'ru', 'es']
    },
    "xai/grok-beta": {
        "max_tokens": 4096,  # Update when official specs are available
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000005,  # Update when pricing is announced
        "output_cost_per_token": 0.000015,  # Update when pricing is announced
        "provider": "xai",
        "mode": "chat",
        "supported_languages_source": "",
        "supported_languages": ['ar', 'zh', 'en', 'fr', 'ru', 'es']
    },
    "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo": {
        "max_tokens": 4096,
        "max_input_tokens": 32768,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0,  # Update when pricing is available
        "output_cost_per_token": 0,  # Update when pricing is available
        "litellm_provider": "together_ai",
        "mode": "chat",
        "supported_languages_source": "https://qwenlm.github.io/blog/qwen2.5/",
        "supported_languages": ['ar', 'zh', 'en', 'fr', 'ru', 'es']
    },
    "together_ai/Qwen/Qwen2.5-14B-Instruct-Turbo": {
        "max_tokens": 4096,
        "max_input_tokens": 32768,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0,  # Update when pricing is available
        "output_cost_per_token": 0,  # Update when pricing is available
        "litellm_provider": "together_ai",
        "mode": "chat",
        "supported_languages_source": "https://qwenlm.github.io/blog/qwen2.5/",
        "supported_languages": ['ar', 'zh', 'en', 'fr', 'ru', 'es']
    },
    "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo": {
        "max_tokens": 4096,
        "max_input_tokens": 32768,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0,  # Update when pricing is available
        "output_cost_per_token": 0,  # Update when pricing is available
        "litellm_provider": "together_ai",
        "mode": "chat",
        "supported_languages_source": "https://qwenlm.github.io/blog/qwen2.5/",
        "supported_languages": ['ar', 'zh', 'en', 'fr', 'ru', 'es']
    },
    "together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {
        "max_tokens": 4096,
        "max_input_tokens": 16384,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0,  # Update when pricing is available
        "output_cost_per_token": 0,  # Update when pricing is available
        "litellm_provider": "together_ai",
        "mode": "chat",
        "supported_languages": ['en', 'fr', 'es']
    },
    "together_ai/meta-llama/Llama-3.2-3B-Instruct-Turbo": {
        "max_tokens": 4096,
        "max_input_tokens": 16384,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0,  # Update when pricing is available
        "output_cost_per_token": 0,  # Update when pricing is available
        "litellm_provider": "together_ai",
        "mode": "chat",
        "supported_languages_source": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md",
        "supported_languages": ['en', 'fr', 'es']
    },
    "together_ai/meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": {
        "max_tokens": 4096,
        "max_input_tokens": 16384,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0,  # Update when pricing is available
        "output_cost_per_token": 0,  # Update when pricing is available
        "litellm_provider": "together_ai",
        "mode": "chat",
        "supported_languages_source": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md",
        "supported_languages": ['en', 'fr', 'es']
    },
    "together_ai/meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": {
        "max_tokens": 4096,
        "max_input_tokens": 16384,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0,  # Update when pricing is available
        "output_cost_per_token": 0,  # Update when pricing is available
        "litellm_provider": "together_ai",
        "mode": "chat",
        "supported_languages_source": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md",
        "supported_languages": ['en', 'fr', 'es']
    },
    "gemini/gemini-exp-1114": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0,  # Update when pricing is available
        "output_cost_per_token": 0,  # Update when pricing is available
        "litellm_provider": "gemini",
        "mode": "chat",
        "supported_languages_source": "https://cloud.google.com/gemini/docs/codeassist/supported-languages",
        "supported_languages": ['ar', 'zh', 'en', 'fr', 'ru', 'es']
    },
    "deepseek/deepseek-chat": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000014,
        "input_cost_per_token_cache_hit": 0.000000014,
        "output_cost_per_token": 0.00000028,
        "litellm_provider": "deepseek",
        "mode": "chat",
    },
    "ai21/jamba-1.5-mini": {
        "max_tokens": 256000,
        "max_input_tokens": 256000,
        "max_output_tokens": 256000,
        "input_cost_per_token": 0.0000002,
        "output_cost_per_token": 0.0000008,
        "litellm_provider": "ai21",
        "mode": "chat",
        "supported_languages_source": "https://docs.ai21.com/docs/jamba-15-models",
        "supported_languages": ['ar', 'en', 'fr', 'es']
    },
    "ai21/jamba-1.5-large": {
        "max_tokens": 256000,
        "max_input_tokens": 256000,
        "max_output_tokens": 256000,
        "input_cost_per_token": 0.000002,
        "output_cost_per_token": 0.000008,
        "litellm_provider": "ai21",
        "mode": "chat",
        "supported_languages_source": "https://docs.ai21.com/docs/jamba-15-models",
        "supported_languages": ['ar', 'en', 'fr', 'es']
    }
}

LANGCHAIN_LLMS_METADATA_DICT = {
    "ERNIE-Bot": {
        "supported_languages_source": "https://fastbots.ai/blog/what-is-ernie-bot-introducing-the-latest-ai-technology",
        "supported_languages": ['zh', 'en']  # Testing is only done in zh and en, but they seem to offer more languages in their tools
    },
    "YandexGPT": {
        "supported_languages_source": "https://ya.ru/ai/gpt-4",
        "supported_languages": ['ru', 'en']
    }
}

OPENAI_LLMS_METADATA_DICT = {
    "xai/grok-beta": {
        "max_tokens": 4096,  # Update when official specs are available
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000005,  # Update when pricing is announced
        "output_cost_per_token": 0.000015,  # Update when pricing is announced
        "provider": "xai",
        "mode": "chat",
        "supported_languages_source": "",
        "supported_languages": ['ar', 'zh', 'en', 'fr', 'ru', 'es']
    },
}


LOCAL_LLM_METADATA_DICT = {
    "jais": {
        "supported_languages_source": "https://huggingface.co/inceptionai/jais-family-30b-8k-chat",
        "supported_languages": ['ar',  'en']
    },
    # "jais_no_system_prompt": {
    #     "supported_languages_source": "https://huggingface.co/inceptionai/jais-13b",
    #     "supported_languages": ['ar',  'en']
    # },
    "Vikhr-Nemo": {
        "supported_languages": ['ru',  'en']
    },
    "silma": {
        "supported_languages": ['ar',  'en']
    },
    "bainchun": {
        "supported_languages": ['zh',  'en']
    },
    "teuken": {
        "supported_languages": ['fr', 'en', 'es']
    },
    "euro-llm": {
        "supported_languages": ['ar', 'zh', 'en', 'fr', 'ru', 'es']
    }
}