import json
import openai
import string
import argparse
import re

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


def gen_prompt_input(ques, rat, ans):
    sys_prompt = "You are a helpful and precise assistant for evaluating the quality of a rationale provided for a retrieved answer to a given question."
    prompt_template = "[Question]\n{ques}\n\n[Retrieved Answer]\n{ans}\n\n[Rationale for Retrieved Answer]\n{rat}\n\n[End of Rationale]\n\n[System]\n{criteria}\n\n"
    criteria = "We would like you to answer several questions related to the quality of the rationale provided for the retrieved answer to the given question. \n" + \
               "1. Why is this rationale not good for the given question and retrieved answer? Analyze based on Relevance, Accuracy, Helpfulness, and Level of Detail. \n" + \
               "2. Based on your analysis, generate a better rationale that provides a more complete and accurate explanation. Format it as: [Better Rationale] your rationale [End]\n"
    prompt = prompt_template.format(
        ques=ques, rat=rat, ans=ans, criteria=criteria
    )
    return sys_prompt, prompt



def extract_segments(text):
    if text.count('[Better Rationale]') >= 2:
        pattern = r'\[(Better Rationale)\](.*?)(\[End\]|\[Better Rationale\]|$)'
        segments = re.findall(pattern, text, re.DOTALL)
    else:
        # pattern = r'\[(Better Answer)\](.*?)\[End\]'
        pattern = r'\[(Better Rationale)\](.*?)(\[End\]|End|$)'
        segments = re.findall(pattern, text, re.DOTALL)
    return [segment[1].strip() for segment in segments]


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
        default=2048,
        help="maximum number of tokens produced in the output",
    )

    args = parser.parse_args()
    openai.api_key = args.api_key
    azure_openai=openai.AzureOpenAI(
        api_key="48XbXJ1j5u1gACCfQ8ajMunUpBdbey7nlIg6a2F8xpVK6ftZqSNUJQQJ99ALAC77bzfXJ3w3AAABACOGPqC1",
        azure_endpoint="https://flowbee-openai-gpt4o-mini-dec6.openai.azure.com/",
        api_version="2024-05-01-preview"
    )
    model_engine = args.api_model

    with open(args.raw_data_path,'r') as f:
        raw_data = json.load(f)

    with open(args.ori_data_path,'r') as f:
        ori_data = json.load(f)

    new_data = []
    for i, raw_data_i in enumerate(raw_data):
        if (i+1) % 1000 == 0:
            print(i+1,'/',len(raw_data))
        seg_list = extract_segments(raw_data_i)

        ori_data_i = ori_data[i]
        ques_i = ori_data_i['question'].strip()
        rat_i = ori_data_i['rationale'].strip()
        ans_i = ori_data_i['answer'].strip()
        if 'input' in ori_data_i.keys():
            input_i = ori_data_i['input'].strip()
        else:
            input_i = ''

        if len(seg_list) != 1:

            if input_i == '':
                sys_prompt, prompt = gen_prompt_no_input(ques_i, rat_i)
            else:
                sys_prompt, prompt = gen_prompt_input(ques_i, rat_i, ans_i)
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
                #             max_tokens=2048,
                #             top_p=1.0,
                # )
                # response = completion.choices[0].message.content
                completion2= azure_openai.chat.completions.create(
                    model='gpt-4o-mini-2',
                    messages=message,
                    temperature=0.0,
                    max_tokens=2048,
                    top_p=1.0,
                    stream=False
                )
                response = completion2.choices[0].message.content
            except:
                response = ''

            seg_list = extract_segments(response)
            pass

        if len(seg_list) != 1:
            seg_list = ['']
    
        temp_data = {}
        temp_data['question'] = ori_data_i['question']
        temp_data['rationale'] = ori_data_i['rationale']
        temp_data['input'] = input_i
        temp_data['better_rationale'] = seg_list[0]
        new_data.append(temp_data)


    if args.save_intermediate_path != '':
        with open(args.save_intermediate_path,'w') as f:
            json.dump(new_data,f,indent=4)

    final_new_data = []
    none_count = 0
    for i, data_i in enumerate(new_data):
        
        temp_data = {}
        temp_data['question'] = data_i['question']
        temp_data['input'] = data_i['input']

        if data_i['better_rationale'] == '':
            none_count += 1
            temp_data['rationale'] = data_i['rationale']
        else:
            temp_data['rationale'] = data_i['better_rationale']

    print('none_num',none_count)
    print('Len New Data', len(final_new_data))
    with open(args.save_path,'w') as f:
        json.dump(final_new_data,f,indent=4)