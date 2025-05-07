import json
import openai
from openai import AzureOpenAI
import string
import argparse
import re

import tiktoken
gpt_encoder = tiktoken.get_encoding("cl100k_base")

# def gen_prompt_no_input(ques, rat, ctx, ans):
def gen_prompt_no_input(ques, rat, ans):

    sys_prompt = "You are a helpful, precise but picky assistant for checking the quality of a given instruction."
    # prompt_template = "[Question]\n{ques}\n\n[Rationale for Retrieved Answer]\n{rat}\n\n[End of Rationale]\n\n[Context]\n{ctx}\n\n[End of Context]\n\n[Answer]\n{ans}\n\n[End of Answer]\n\n[System]\n{criteria}\n\n"
    prompt_template = "[Question]\n{ques}\n\n[Rationale for Retrieved Answer]\n{rat}\n\n[End of Rationale]\n\n[Answer]\n{ans}\n\n[End of Answer]\n\n[System]\n{criteria}\n\n"
    criteria = "We would like you to answer several questions related to the quality of a given question and the rationale for the answer retrieved by a system. \n" + \
            "1. Why is this question potentially problematic? Evaluate it based on: Ambiguity, Required Knowledge, Context Clarity, and Answerability. \n" + \
            "Then evaluate why this rationale is not good for the given question-answer pair. Analyze based on: Relevance to the question, Accuracy in explaining the retrieval, Helpfulness, and Level of Detail. \n" + \
            "Finally, explain how a poorly formed question might lead to a poor rationale or poor answer justification. \n" + \
            "2. Based on your analysis, generate a new, challenging question that is complex, unambiguous, and requires nuanced reasoning. Ensure it's independent of the original. Format it as: [New Question] your question [End]\n" + \
            "3. Provide a detailed rationale for the answer to this new question (simulate what a good RAG justification should look like). Format it as: [New Rationale] your rationale [End] \n"

    prompt = prompt_template.format(
        ques=ques,
        rat=rat,
        # ctx=ctx,
        ans=ans,
        criteria=criteria
    )
    return sys_prompt, prompt



def extract_ques(text,no_input=True):
    if '[New Question]' in text:
        pattern = r'(\[New Question\])(.*?)(\[End\]|\[New Rationale\]|New Rationale:)'
    else:
        pattern = r'(New Question:)(.*?)(\[End\]|\[New Rationale\]|New Rationale:)'
    segments = re.findall(pattern, text, re.DOTALL)
    if len(segments) == 0:
        seg_ins = ''
    else:
        seg_ins = segments[0][1].strip()
    if seg_ins.endswith("\n\n3."):
        seg_ins = seg_ins[:-4]
    return seg_ins

def extract_rat(text,no_input=True):
    if '[New Rationale]' in text:
        pattern = r'(\[New Rationale\])(.*?)(\[End\]|$)'
    else:
        pattern = r'(New Rationale:)(.*?)(\[End\]|$)'
        # pattern = r'(\[New Answer\]|New Answer:)(.*?)(\[End\]|$)'
    segments = re.findall(pattern, text, re.DOTALL)
    if len(segments) == 0:
        seg_oup = ''
    else:
        seg_oup = segments[0][1].strip()
    return seg_oup

def extract_segments_no_input(text):
    if text == '':
        return []
    seg_ins = extract_ques(text,no_input=True)
    seg_oup = extract_rat(text,no_input=True)
    return [seg_ins,seg_oup]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str, default='')
    parser.add_argument("--ori_data_path", type=str, default='')
    parser.add_argument("--save_path", type=str, default='')
    parser.add_argument("--save_intermediate_path", type=str, default='')
    parser.add_argument("--api_key", type=str, default='')
    parser.add_argument("--api_model",type=str,default='gpt-3.5-turbo')
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8008,
        help="maximum number of tokens produced in the output",
    )

    args = parser.parse_args()
    openai.api_key = args.api_key
    azure_openai=AzureOpenAI(
        api_key="48XbXJ1j5u1gACCfQ8ajMunUpBdbey7nlIg6a2F8xpVK6ftZqSNUJQQJ99ALAC77bzfXJ3w3AAABACOGPqC1",
        azure_endpoint="https://flowbee-openai-gpt4o-mini-dec6.openai.azure.com/",
        api_version="2024-05-01-preview"
    )
    model_engine = args.api_model

    raw_json_data_path = args.raw_data_path
    with open(raw_json_data_path,'r') as f:
        raw_data = json.load(f)

    ori_json_data_path = args.ori_data_path
    with open(ori_json_data_path,'r') as f:
        ori_data = json.load(f)

    new_data = []
    retry_num = 0
    for i, raw_data_i in enumerate(raw_data):

        if (i+1) % 1000 == 0:
            print(i+1,'/',len(raw_data))

        ori_data_i = ori_data[i]
        ques_i = ori_data_i['question'].strip()
        rat_i = ori_data_i['rationale'].strip()
        ans_i = ori_data_i['answer'].strip()
        if 'input' in ori_data_i.keys():
            input_i = ori_data_i['input'].strip()
        else:
            ori_data_i['input'] = ''
            input_i = ''
            
        retry_flag = False
        seg_list = extract_segments_no_input(raw_data_i)
        if len(seg_list) != 2:
            retry_flag = True
        else:
            if seg_list[0] == '' and seg_list[1] == '':
                retry_flag = True
            if (seg_list[0] == '') or ('your question' in seg_list[0]):
                seg_list[0] = ques_i
            if ('N/A' in seg_list[1]) or (seg_list[1]=='') or ('your rationale' in seg_list[1]):
                seg_list[1] = rat_i

        if retry_flag:
            retry_num += 1

            sys_prompt, prompt = gen_prompt_no_input(ques_i, rat_i, ans_i)
            
            token_limit = min(args.max_tokens,4050-len(gpt_encoder.encode(prompt)))
            response = ''
            try:
                message =[
                            {"role": "system", "content": sys_prompt},
                            {
                                "role": "user",
                                "content": prompt,
                            },
                ]
                # completion = openai.ChatCompletion.create(
                #             model=model_engine,
                #             messages=message,
                #             temperature=0.0,
                #             max_tokens=token_limit,
                #             top_p=1.0,
                # )
                completion2= azure_openai.chat.completions.create(
                    model='gpt-4o-mini-2',
                    messages=message,
                    temperature=0.0,
                    max_tokens=token_limit,
                    top_p=1.0,
                    stream=False
                )
                response = completion2.choices[0].message.content
            except:
                response = ''

            seg_list = extract_segments_no_input(response)
            # seg_list = [x for x in seg_list if x != '']


        temp_data = {}
        temp_data['question'] = ori_data_i['question']
        temp_data['rationale'] = ori_data_i['rationale']
        temp_data['answer'] = ori_data_i['answer']
        temp_data['input'] = ori_data_i['input']

        if len(seg_list) != 2:
            temp_data['new_question'] = ori_data_i['question']
            temp_data['new_rationale'] = ori_data_i['rationale']
        else:
            if (seg_list[0] == '') or ('your question' in seg_list[0]):
                temp_data['new_question'] = ori_data_i['question']
            else:
                temp_data['new_question'] = seg_list[0]

            if ('N/A' in seg_list[1]) or (seg_list[1]=='') or ('your rationale' in seg_list[1]):
                temp_data['new_rationale'] = ori_data_i['rationale']
            else:
                temp_data['new_rationale'] = seg_list[1]

        temp_data['new_input'] = ''
        new_data.append(temp_data)

        pass
    print('retry_num',retry_num)
    if args.save_intermediate_path != '':
        with open(args.save_intermediate_path,'w') as f:
            json.dump(new_data,f,indent=4)
    
    final_new_data = []
    none_count = 0
    for i, data_i in enumerate(new_data):
        temp_data = {}

        if (data_i['new_question'] == '') and (data_i['new_rationale'] == ''):
            none_count += 1
            temp_data['question'] = data_i['question']
            temp_data['rationale'] = data_i['rationale']
            temp_data['answer'] = data_i['answer']
            temp_data['input'] = data_i['input']
        else:
            temp_data['question'] = data_i['new_question']
            temp_data['rationale'] = data_i['new_rationale']
            temp_data['answer'] = data_i['answer']
            temp_data['input'] = data_i['new_input'] 

        final_new_data.append(temp_data)

    print('none_num',none_count)
    print('Len New Data', len(final_new_data))
    with open(args.save_path,'w') as f:
        json.dump(final_new_data,f,indent=4)