import transformers
import torch

model_id = "/ssd-data1/cxf2022/10.LLaMA/Qwen2.5-3B-Instruct"

# 加载配置并手动设置parallel_style
config = transformers.AutoConfig.from_pretrained(model_id)
config.parallel_style = "none"  # 或者 "ddp" 或 "fsdp"

# 加载模型时传入修改后的配置
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    config=config,  # 直接传入配置
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])