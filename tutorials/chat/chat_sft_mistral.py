import os
from datasets import DatasetDict
from datasets import load_dataset

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['NCCL_P2P_DISABLE'] = "1"
os.environ['NCCL_IB_DISABLE'] = '1'

raw_datasets = load_dataset("HuggingFaceH4/ultrachat_200k")
raw_datasets = DatasetDict({
    "train": raw_datasets["train_sft"].select(range(10000)),
    "test": raw_datasets["test_sft"].select(range(10000))
})


from transformers import AutoTokenizer

model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# set pad_token_id equal to the eos_token_id if not set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Set reasonable default for models without max length
if tokenizer.model_max_length > 100_000:
    tokenizer.model_max_length = 1024

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

import re
import random
from multiprocessing import cpu_count

def apply_chat_template(example, tokenizer):
    messages = example["messages"]
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

    return example

column_names = list(raw_datasets["train"].features)


raw_datasets = raw_datasets.map(apply_chat_template,
                                num_proc=cpu_count(),
                                fn_kwargs={"tokenizer": tokenizer},
                                remove_columns=column_names,
                                desc="Applying chat template",)

# create the splits
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]


from transformers import BitsAndBytesConfig, TrainingArguments
import torch
from transformers import AutoModelForCausalLM
from trl import SFTConfig

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_compute_dtype=torch.bfloat16,
)
device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

model_kwargs = dict(
    attn_implementation="flash_attention_2", 
    torch_dtype="auto",
    use_cache=False, # set to False as we're going to use gradient checkpointing
    device_map=device_map,
    quantization_config=quantization_config,
)


model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)


from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments

# path where the Trainer will save its checkpoints and logs
output_dir = 'data/mistral-7b-sft-lora'

# based on config
training_args = SFTConfig(
    fp16=True, # specify bf16=True instead when training on GPUs that support bf16
    do_eval=True,
    eval_strategy="epoch",
    per_device_eval_batch_size=4, # originally set to 8
    per_device_train_batch_size=2, # originally set to 8
    gradient_accumulation_steps=64,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2.0e-05,
    log_level="info",
    logging_steps=5,
    logging_strategy="steps",
    lr_scheduler_type="cosine",
    max_steps=-1,
    num_train_epochs=1,
    output_dir=output_dir,
    overwrite_output_dir=True,
    report_to="wandb",
    save_strategy="no",
    save_total_limit=None,
    seed=42,
    dataset_text_field="text",
    packing=True,
    max_seq_length=tokenizer.model_max_length,
    dataset_num_proc=cpu_count(),
)

# based on config
peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

trainer = SFTTrainer(
        # model=model_id,
        # model_init_kwargs=model_kwargs,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

train_result = trainer.train()

metrics = train_result.metrics
max_train_samples = training_args.max_train_samples if training_args.max_train_samples is not None else len(train_dataset)
metrics["train_samples"] = min(max_train_samples, len(train_dataset))
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()


