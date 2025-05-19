import os
import asyncio
import json
from typing import List, Dict, Any, Optional
import aiohttp
import time
import logging

logger = logging.getLogger(__name__)


class AzureOpenAIClient:
    """Client for interacting with Azure OpenAI API with batch processing capabilities."""

    def __init__(
        self,
        api_key: str,
        endpoint_url: str,
        api_version: str,
        deployment_name: str,
        max_concurrent_requests: int = 200,
        max_tokens: int = 4096,
        temperature: float = 0,
        timeout: int = 300,
    ):
        """Initialize the Azure OpenAI client.

        Args:
            api_key: Azure OpenAI API key
            endpoint_url: Azure OpenAI endpoint URL
            api_version: Azure OpenAI API version
            deployment_name: Azure OpenAI deployment name
            max_concurrent_requests: Maximum number of concurrent requests
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            timeout: Timeout for requests in seconds
        """
        self.api_key = api_key
        self.endpoint_url = endpoint_url.rstrip("/")
        self.api_version = api_version
        self.deployment_name = deployment_name
        self.max_concurrent_requests = max_concurrent_requests
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self.url = f"{self.endpoint_url}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        # Semaphore to control concurrency
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def _generate_completion(
        self, prompt: str, session: aiohttp.ClientSession
    ) -> Dict[str, Any]:
        """Generate a completion for a single prompt."""
        async with self.semaphore:
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

            retry_count = 0
            max_retries = 3
            backoff_time = 1

            while retry_count < max_retries:
                try:
                    async with session.post(
                        self.url,
                        headers=self.headers,
                        json=payload,
                        timeout=self.timeout,
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return {
                                "prompt": prompt,
                                "output": result["choices"][0]["message"]["content"],
                                "status": "success",
                            }
                        else:
                            error_text = await response.text()
                            logger.warning(
                                f"Request failed with status {response.status}: {error_text}"
                            )

                            # Rate limiting or server errors should be retried
                            if response.status == 429 or response.status >= 500:
                                retry_count += 1
                                await asyncio.sleep(backoff_time)
                                backoff_time *= 2  # Exponential backoff
                                continue

                            return {
                                "prompt": prompt,
                                "output": "",
                                "status": "failed",
                                "error": f"HTTP {response.status}: {error_text}",
                            }
                except Exception as e:
                    logger.warning(f"Request exception: {str(e)}")
                    retry_count += 1
                    if retry_count < max_retries:
                        await asyncio.sleep(backoff_time)
                        backoff_time *= 2
                    else:
                        return {
                            "prompt": prompt,
                            "output": "",
                            "status": "failed",
                            "error": str(e),
                        }

    async def generate_completions(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Generate completions for multiple prompts in parallel."""
        async with aiohttp.ClientSession() as session:
            tasks = [self._generate_completion(prompt, session) for prompt in prompts]
            return await asyncio.gather(*tasks)

    def batch_generate_completions(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Generate completions for multiple prompts in batch mode."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.generate_completions(prompts))


class AzureOpenAIOutput:
    """Class to store Azure OpenAI outputs and format them like vLLM outputs."""

    def __init__(self, prompt: str, generated_text: str):
        self.prompt = prompt
        self.outputs = [self._OutputItem(generated_text)]

    class _OutputItem:
        def __init__(self, text: str):
            self.text = text


def generate_rationale_with_azure(
    data_list: List[Dict[str, Any]],
    prompts: List[str],
    output_file: str,
    n_docs: int,
    api_key: str,
    endpoint_url: str,
    api_version: str,
    deployment_name: str,
    max_concurrent_requests: int = 200,
    max_tokens: int = 4096,
    temperature: float = 0,
) -> List[Dict[str, Any]]:
    """Generate rationales using Azure OpenAI API.

    Args:
        data_list: List of data items
        prompts: List of prompts
        output_file: Output file path
        n_docs: Number of retrieved documents
        api_key: Azure OpenAI API key
        endpoint_url: Azure OpenAI endpoint URL
        api_version: Azure OpenAI API version
        deployment_name: Azure OpenAI deployment name
        max_concurrent_requests: Maximum number of concurrent requests
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling

    Returns:
        List of outputs with rationales added
    """
    start_time = time.time()
    logger.info(
        f"Generating rationales for {len(prompts)} examples using Azure OpenAI..."
    )

    client = AzureOpenAIClient(
        api_key=api_key,
        endpoint_url=endpoint_url,
        api_version=api_version,
        deployment_name=deployment_name,
        max_concurrent_requests=max_concurrent_requests,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    results = client.batch_generate_completions(prompts)

    # Convert to the format expected by save_outputs
    azure_outputs = []
    for i, result in enumerate(results):
        if result["status"] == "success":
            azure_outputs.append(AzureOpenAIOutput(prompts[i], result["output"]))
        else:
            # Handle failed requests with empty output
            logger.warning(f"Failed request: {result.get('error', 'Unknown error')}")
            azure_outputs.append(AzureOpenAIOutput(prompts[i], ""))

    elapsed_time = time.time() - start_time
    logger.info(
        f"Generated {len(azure_outputs)} rationales in {elapsed_time:.2f} seconds"
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return azure_outputs
