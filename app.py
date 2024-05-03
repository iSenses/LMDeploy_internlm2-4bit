import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

base_path = './4bit-1_8b'
os.system(f'git clone https://code.openxlab.org.cn/mingyanglee/internlm2-chat-1_8b-4bit.git {base_path}')

tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.8,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="LMDeploy 1.8B 4bit量化， 来试试呀",
                description="""
4bit quantization on InternLM2 1.8B mainly developed by Shanghai AI Laboratory.  
                 """,
                 ).queue(1).launch()
