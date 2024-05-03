import os
import sys
from lmdeploy.serve.gradio.turbomind_coupled import run_local
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

backend_config = TurbomindEngineConfig(cache_max_entry_count=0.2)
chat_template_config = ChatTemplateConfig(model_name='internlm2-chat-1_8b')

base_path = './4bit-1_8b'
os.system(f'git clone https://code.openxlab.org.cn/mingyanglee/internlm2-chat-1_8b-4bit.git {base_path}')


run_local(base_path, model_name='internlm2-chat-1_8b', backend_config=backend_config, chat_template_config=chat_template_config, server_port=7860, tp=1)
