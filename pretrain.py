import os
import random
import numpy as np

import datasets
from datasets import load_from_disk, concatenate_datasets
import transformers
from transformers import DataCollatorForLanguageModeling
import evaluate

from nar_transformers import BertTokenizerFast
from nar_transformers import BertConfig
from nar_transformers import EncoderDecoderConfig, EncoderDecoderModel
from nar_transformers import TrainingArguments, Trainer

import wandb
os.environ["WANDB_PROJECT"] = "pre-training"

# Set random seed for permutaion pre-processing
random.seed(42)

#model_name = "bert-base-cased"
#cached_data_dir = "/scratch/aae15163zd/cache/wikipedia-20220301en-bert-base-cased-512/"
model_name = "dmis-lab/biobert-base-cased-v1.1"
cached_data_dir = "/scratch/aae15163zd/cache/pubmed-biobert-base-cased-v1.1-512-plm0.30/"

if cached_data_dir is not None:
    with open(os.path.join(cached_data_dir, "model_name.txt"), "r") as f:
        assert model_name == f.read()

batch_size = 20
max_length = 512
latent_size = 8
pretraining_strategy = "plm" # ae: AutoEncoding, mlm: Masked Language Modeling, plm: Permutation Language Modeling
replace_probability = 0.3

tokenizer = BertTokenizerFast.from_pretrained(model_name)

def process_permutation_language_model_inputs(batch):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    bs = len(input_ids)
    seq_length = attention_mask.sum(-1) - 2 # Excluding [CLS] and [SEP] tokens
    num_permuted_tokens = (seq_length * replace_probability).int()

    permuted_input_ids = input_ids.clone()
    for i in range(bs):
        target_indices = random.sample(range(1, seq_length[i]+1), num_permuted_tokens[i])
        permuted_indices = random.sample(target_indices, len(target_indices))
        permuted_input_ids[i, np.array(target_indices, dtype="int")] = input_ids[i, np.array(permuted_indices, dtype="int")]
    batch["input_ids"] = permuted_input_ids

    return batch

train_data = load_from_disk(os.path.join(cached_data_dir, "train"))
val_data = load_from_disk(os.path.join(cached_data_dir, "valid"))
val_data = val_data.select(range(5000))

tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name, latent_size)
model.config.is_vae = False
model.config.dropout_prob = 0.0

# set special tokens
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id


#run_name = "pubmed-biobert-base-cased-plm0.30-lr5e-5"
run_name = "pubmed-biobert-base-cased-plm0.30-lr1e-4-bs2560"
# set training arguments - these params are not really tuned, feel free to change
training_args = TrainingArguments(
    output_dir=os.path.join("~/my_data/pretraining", run_name),
    evaluation_strategy="steps",
    save_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    logging_steps=1_000,  # set to 1000 for full training
    save_steps=5_000,  # set to 500 for full training
    eval_steps=1_000,  # set to 8000 for full training
    warmup_ratio=0.05,  # set to 2000 for full training
    learning_rate=1e-04,
    weight_decay=0.01,
    num_train_epochs=5.0, # seems like the default is only 3.0
    overwrite_output_dir=True,
    save_total_limit=None,
    bf16=True,
    torch_compile=True,
    report_to="wandb",
    run_name=run_name,
    gradient_accumulation_steps=8,
)

if pretraining_strategy == "mlm":
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=replace_probability,
    )
else:
    data_collator = None

def compute_metrics(pred):
    return {}

# instantiate trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
)

trainer.train()
#trainer.train(resume_from_checkpoint=True)
