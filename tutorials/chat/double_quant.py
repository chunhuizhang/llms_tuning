import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

import argparse

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

parser = argparse.ArgumentParser(description='Quantize a model using bitsandbytes')

parser.add_argument('-double', '--bnb_4bit_use_double_quant', action='store_true', default=False,
                    help='Whether to use double quantization for 4-bit quantization')
args = parser.parse_args()


# 配置量化参数
quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_compute_dtype=torch.float16
)

# 加载模型前记录内存
m0 = torch.cuda.memory_allocated()
print(m0/(1024*1024*1024))

model_id = 'mistralai/Mistral-7B-v0.1'
# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
   model_id,
   quantization_config=quantization_config,
   device_map="auto"
)

# 加载模型后记录内存
print((torch.cuda.memory_allocated() - m0)/(1024*1024*1024))

# 检查模型配置
print(model.config)