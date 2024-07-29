import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['WANDB_DISABLED'] = 'true'


from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
# from accelerate import Accelerator

import torch
torch.manual_seed(42)

dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")

model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", 
                                            #  device_map={"": Accelerator().process_index}
                                            # device_map={"": 0}
                                             )
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

args = SFTConfig(output_dir="/tmp", 
                 max_seq_length=512, 
                 num_train_epochs=2, 
                 per_device_train_batch_size=8, 
                 gradient_accumulation_steps=4,
                 gradient_checkpointing=True,
                 bf16=True
                 )

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=args,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()
