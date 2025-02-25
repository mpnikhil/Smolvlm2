import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

USE_LORA = False
USE_QLORA = False
SMOL = True

model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" if SMOL else "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

torch.set_default_dtype(torch.float32)

processor = AutoProcessor.from_pretrained(
    model_id,
    torch_dtype=torch.float32
)

if USE_QLORA or USE_LORA:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
        use_dora=False if USE_QLORA else True,
        init_lora_weights="gaussian"
    )
    lora_config.inference_mode = False
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config if USE_QLORA else None,
        _attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model.add_adapter(lora_config)
    model.enable_adapters()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    print(model.get_nb_trainable_parameters())
else:
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    ).to(device)

peak_mem = torch.cuda.max_memory_allocated()
print(f"The model as is is holding: {peak_mem / 1024**3:.2f} of GPU RAM")

if "<image>" in processor.tokenizer.get_vocab():
    print("✅ `<image>` token already present.")
else:
    print("🚨 `<image>` token not found. Adding it...")
    processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
    model.resize_token_embeddings(len(processor.tokenizer))

from datasets import load_dataset
ds = load_dataset('mpnikhil/kitchen-classifier', trust_remote_code=True)

train_ds = ds["train"]
eval_ds = ds["validation"]

train_ds

print(ds["train"][0])

image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")]

def collate_fn(examples):
  texts = []
  images = []
  for example in examples:
      image = example["image"]
      if image.mode != 'RGB':
        image = image.convert('RGB')
      question = example["question"]
      answer = example["answer"]
      messages = [
          {
              "role": "user",
              "content": [
                  {"type": "text", "text": "Answer briefly."},
                  {"type": "image"},
                  {"type": "text", "text": question}
              ]
          },
          {
              "role": "assistant",
              "content": [
                  {"type": "text", "text": answer}
              ]
          }
      ]
      text = processor.apply_chat_template(messages, add_generation_prompt=False)
      texts.append(text.strip())
      images.append([image])

  batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
  batch["input_ids"] = batch["input_ids"].to(torch.long)  # input_ids must be integers
  for k in batch:
    if isinstance(batch[k], torch.Tensor) and k != "input_ids":
        batch[k] = batch[k].to(torch.float32)
  labels = batch["input_ids"].clone()
  labels[labels == processor.tokenizer.pad_token_id] = -100
  labels[labels == image_token_id] = -100
  batch["labels"] = labels

  return batch

from transformers import TrainingArguments, Trainer

model_name = model_id.split("/")[-1]

training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=1,
    optim="adamw_torch", # for 8-bit, keep this, else adamw_hf
    output_dir=f"./{model_name}-mpnikhil1",
    hub_model_id=f"{model_name}-mpnikhil1",
    report_to="tensorboard",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    use_mps_device=True,
    dataloader_pin_memory=False if device.type == "mps" else True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,

  )

trainer.train()

trainer.push_to_hub()