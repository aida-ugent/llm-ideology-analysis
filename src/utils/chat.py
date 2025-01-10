import asyncio
import ssl

import aiohttp
import certifi
import requests
from tqdm.auto import tqdm


class ChatCompletor:
    def __init__(self, apikey, model="gpt-3.5-turbo", max_tokens=2048, temperature=0., chunk_size=100):
        self._url = "https://api.openai.com/v1/chat/completions"
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {apikey}",
        }
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._chunk_size = chunk_size

    def _get_payload(self, messages: str) -> dict:
        return {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }

    async def call_async(self, session, target: str) -> str:
        try:
            async with session.post(
                url=self._url,
                headers=self._headers,
                json=self._get_payload(target),
                ssl=ssl.create_default_context(cafile=certifi.where()),
            ) as response:
                response = await asyncio.wait_for(response.json(), timeout=80)

            if "error" in response:
                print(f"OpenAI request failed with error {response['error']}")
            return response["choices"][0]["message"]["content"]

        except asyncio.TimeoutError:
            print("The request has timed out.")
        except Exception as e:
            print("Request failed: ", str(e))

    async def call_chatgpt_bulk(self, prompts) -> list[str]:
        async with aiohttp.ClientSession() as session:
            responses = []
            with tqdm(total=len(prompts), desc="Translating") as pbar:
                for i in range(0, len(prompts), self._chunk_size):
                    end = (
                        i + self._chunk_size
                        if (i + self._chunk_size < len(prompts))
                        else len(prompts)
                    )
                    prompts_chunk = prompts[i:end]
                    chunk_responses = await asyncio.gather(
                        *[self.call_async(session, prompt) for prompt in prompts_chunk]
                    )
                    responses += chunk_responses
                    pbar.update(len(chunk_responses))
        return responses

    def call_chatgpt_sync(self, target: str) -> str:
        try:
            response = requests.post(
                url=self._url,
                headers=self._headers,
                json=self._get_payload(target),
                verify=certifi.where(),
            )
            response = response.json()
            if "error" in response:
                print(f"OpenAI request failed with error {response['error']}")
            return response["choices"][0]["message"]["content"]
        except:
            print("Request failed.")
