import argparse
import json
import os
import time

import openai
from tqdm import tqdm
import asyncio
from typing import Any
import logging
from typing import List, Dict, Any

from pyserini.search.lucene import LuceneSearcher

import tiktoken
gpt_encoder = tiktoken.get_encoding("cl100k_base")
from transformers import (
    LlamaForCausalLM, LlamaTokenizer)
llama_tokenizer = LlamaTokenizer.from_pretrained('baffo32/decapoda-research-llama-7B-hf')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def dispatch_openai_requests(
    messages_list: List[List[Dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> List[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.

    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        # openai.ChatCompletion.acreate(
        #     model=model,
        #     messages=x,
        #     temperature=temperature,
        #     max_tokens=max_tokens,
        #     top_p=top_p,
        # )

        azure_openai.chat.completions.create(
            model='gpt-4o-mini-2',
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

def gen_prompt_no_input(ques, rat):
    sys_prompt = "You are a helpful, precise but picky assistant for evaluating the quality of a rationale provided for a retrieved answer to a given question."
    prompt_template = "[Question]\n{ques}\n\n[Rationale for Retrieved Answer]\n{rat}\n\n[End of Rationale]\n\n[System]\n{criteria}\n\n"
    criteria = "We would like you to answer several questions related to the quality of the rationale provided for a retrieved answer to the given question. \n" + \
               "1. Why is this rationale not good for the given question? Analyze based on Relevance to the question, Accuracy of the explanation, Helpfulness, and Level of Detail. \n" + \
               "2. Based on your analysis, generate a better rationale that clearly explains why the retrieved answer is suitable. Format it as: [Better Rationale] your rationale [End]\n"
    prompt = prompt_template.format(
        ques=ques, rat=rat, criteria=criteria
    )
    return sys_prompt, prompt


def gen_prompt_input(ques, rat, ctx, ans):
    sys_prompt = "You are a helpful and precise assistant for evaluating the quality of a rationale provided for a retrieved answer to a given question."
    prompt_template = "[Question]\n{ques}\n\n[Retrieved Answer]\n{ans}\n\n[Rationale for Retrieved Answer]\n{rat}\n\n[End of Rationale]\n\n[System]\n{criteria}\n\n"
    criteria = "We would like you to answer several questions related to the quality of the rationale provided for the retrieved answer to the given question. \n" + \
               "1. Why is this rationale not good for the given question and retrieved answer? Analyze based on Relevance, Accuracy, Helpfulness, and Level of Detail. \n" + \
               "2. Based on your analysis, generate a better rationale that provides a more complete and accurate explanation. Format it as: [Better Rationale] your rationale [End]\n"
    prompt = prompt_template.format(
        ques=ques, rat=rat, ctx=ctx, ans=ans, criteria=criteria
    )
    return sys_prompt, prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='')
    parser.add_argument("--save_path", type=str, default='')
    parser.add_argument("--api_key", type=str, default='')
    parser.add_argument("--api_model",type=str,default='gpt-3.5-turbo')
    parser.add_argument("--api_base",type=str,default='')
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size to call OpenAI GPT",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8008,
        help="maximum number of tokens produced in the output",
    )
    args = parser.parse_args()
    # if args.api_base != '':
    #     openai.api_base = args.api_base
    openai.api_key = args.api_key
    azure_openai=openai.AsyncAzureOpenAI(
        api_key="48XbXJ1j5u1gACCfQ8ajMunUpBdbey7nlIg6a2F8xpVK6ftZqSNUJQQJ99ALAC77bzfXJ3w3AAABACOGPqC1",
        azure_endpoint="https://flowbee-openai-gpt4o-mini-dec6.openai.azure.com/",
        api_version="2024-05-01-preview"
    )

    with open(args.data_path, "r") as f:
        data = json.load(f)

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(data)
    sampled_data = data[start_idx:end_idx]
    
    message_list = []
    token_len_list = []
    for i, data_i in enumerate(sampled_data):
        ques_i = data_i['question'].strip()
        rat_i = data_i['rationale'].strip()
        ans_i = data_i['answer'].strip()

        searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr-multi')
        hits = searcher.search(ques_i, k=5)
        # Concatenate top-k passages as a single string
        ctx_i = "\n\n".join([hit.raw for hit in hits])

        # if 'input' in data_i.keys():
        #     input_i = data_i['input'].strip()
        # else:
        #     input_i = ''

        whole_text = ques_i + ctx_i + rat_i + ans_i
        inputs = llama_tokenizer(whole_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] > 2048:
            gap = input_ids.shape[1] - 2048
            ans_i = ans_i[:-gap]

        if ctx_i == '':
            sys_prompt, prompt = gen_prompt_no_input(ques_i, rat_i)
        else:
            sys_prompt, prompt = gen_prompt_input(ques_i, rat_i, ctx_i, ans_i)

        message =[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": prompt,
                    },
        ]
        message_list.append(message)
        token_len_list.append(len(gpt_encoder.encode(prompt)))

    predictions = []
    i = 0
    wait_base = 10
    retry = 0
    error = 0
    pbar = tqdm(total=len(message_list))
    batch_size = args.batch_size
    while(i<len(message_list)):
        token_limit_in_current_batch = min(args.max_tokens,4050-max(token_len_list[i:i+batch_size]))
        try:
            batch_predictions = asyncio.run(
                dispatch_openai_requests(
                    messages_list=message_list[i:i+batch_size],
                    model=args.api_model,
                    temperature=0.0,
                    max_tokens=token_limit_in_current_batch,
                    top_p=1.0,
                )
            )
            predictions += batch_predictions
            retry = 0
            i += batch_size
            wait_base = 10
            pbar.update(batch_size)
        except:
            retry += 1
            error += 1
            print("Batch error: ",i, i+batch_size)
            print("retry number: ", retry)
            print("error number: ", error)
            time.sleep(wait_base)
            wait_base = wait_base*2
    pbar.close()

    new_data = []
    for idx, prediction in enumerate(predictions):
        review=prediction.choices[0].message.content
        # review = prediction['choices'][0]['message']['content']
        new_data.append(review)
        pass

    print('Len New Data', len(new_data))
    with open(args.save_path,'w') as f:
        json.dump(new_data,f,indent=4)

    pass
