import argparse
import json
import os
import time
import asyncio
from typing import List, Dict, Any
import logging
import tiktoken
from transformers import LlamaTokenizer
import openai
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize tokenizers
gpt_encoder = tiktoken.get_encoding("cl100k_base")
llama_tokenizer = LlamaTokenizer.from_pretrained(
    "baffo32/decapoda-research-llama-7B-hf"
)


async def get_teacher_rationale(
    question: str,
    context: str,
    model: str = "gpt-4",
    temperature: float = 0.0,
    max_tokens: int = 1000,
) -> str:
    """Get rationale from teacher model."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that provides detailed rationales for answering questions based on given context.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a detailed rationale for answering this question based on the context.",
        },
    ]

    response = await openai.ChatCompletion.acreate(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return response.choices[0].message.content


async def reflect_on_rationale(
    question: str,
    rationale: str,
    context: str,
    answer: str,
    model: str = "gpt-4",
    temperature: float = 0.0,
    max_tokens: int = 1000,
) -> Dict[str, str]:
    """Reflect on the rationale and generate improved question and rationale."""
    sys_prompt = "You are a helpful, precise but picky assistant for checking the quality of a given instruction."
    prompt = f"""[Question]
{question}

[Rationale for Retrieved Answer]
{rationale}

[End of Rationale]

[Context]
{context}

[End of Context]

[Answer]
{answer}

[End of Answer]

[System]
We would like you to answer several questions related to the quality of a given question and the rationale for the answer retrieved by a system. 
1. Why is this question potentially problematic? Evaluate it based on: Ambiguity, Required Knowledge, Context Clarity, and Answerability. 
Then evaluate why this rationale is not good for the given question-answer pair. Analyze based on: Relevance to the question, Accuracy in explaining the retrieval, Helpfulness, and Level of Detail. 
Finally, explain how a poorly formed question might lead to a poor rationale or poor answer justification. 
2. Based on your analysis, generate a new, challenging question that is complex, unambiguous, and requires nuanced reasoning. Ensure it's independent of the original. Format it as: [New Question] your question [End]
3. Provide a detailed rationale for the answer to this new question (simulate what a good RAG justification should look like). Format it as: [New Rationale] your rationale [End]
"""

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    response = await openai.ChatCompletion.acreate(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )

    content = response.choices[0].message.content

    # Parse the response to extract new question and rationale
    try:
        new_question = content.split("[New Question]")[1].split("[End]")[0].strip()
        new_rationale = content.split("[New Rationale]")[1].split("[End]")[0].strip()
        return {
            "new_question": new_question,
            "new_rationale": new_rationale,
            "reflection": content,
        }
    except:
        logger.error(f"Failed to parse reflection response: {content}")
        return None


async def process_batch(
    batch: List[Dict],
    model: str,
    temperature: float,
    max_tokens: int,
) -> List[Dict]:
    """Process a batch of examples to generate augmented data."""
    augmented_examples = []

    for example in batch:
        # Get teacher rationale
        context = "\n\n".join(
            [f"[Title]: {c['title']}\n[Content]: {c['text']}" for c in example["ctxs"]]
        )
        teacher_rationale = await get_teacher_rationale(
            question=example["question"],
            context=context,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Reflect on rationale
        reflection = await reflect_on_rationale(
            question=example["question"],
            rationale=teacher_rationale,
            context=context,
            answer=example["answers"][0],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if reflection:
            augmented_example = {
                "question": reflection["new_question"],
                "rationale": reflection["new_rationale"],
                "ctxs": example["ctxs"],
                "answers": example["answers"],
                "metadata": {
                    "original_question": example["question"],
                    "teacher_rationale": teacher_rationale,
                    "reflection": reflection["reflection"],
                },
            }
            augmented_examples.append(augmented_example)

    return augmented_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to input data file"
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save augmented dataset"
    )
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument(
        "--model", type=str, default="gpt-4", help="Model to use for generation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=5, help="Batch size for processing"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1000, help="Max tokens for generation"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for generation"
    )
    args = parser.parse_args()

    # Set up OpenAI
    openai.api_key = args.api_key

    # Load data
    with open(args.data_path, "r") as f:
        data = json.load(f)

    # Process in batches
    augmented_data = []
    for i in tqdm(range(0, len(data), args.batch_size)):
        batch = data[i : i + args.batch_size]
        batch_results = asyncio.run(
            process_batch(
                batch=batch,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        )
        augmented_data.extend(batch_results)

        # Save intermediate results
        with open(args.save_path, "w") as f:
            json.dump(augmented_data, f, indent=2)

        # Sleep to avoid rate limits
        time.sleep(1)

    logger.info(f"Generated {len(augmented_data)} augmented examples")


if __name__ == "__main__":
    main()
