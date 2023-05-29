import os
import datasets
from transformers import BertTokenizerFast


#model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
model_name = "dmis-lab/biobert-base-cased-v1.1"
max_length = 512
tokenizer = BertTokenizerFast.from_pretrained(model_name)

#cached_data_dir = "/scratch/aae15163zd/cache/pubmed-{}-{}".format(model_name.split("/")[-1], max_length)
cached_data_dir = "/scratch/aae15163zd/cache/pubmed-{}-{}-plm0.30".format(model_name.split("/")[-1], max_length)
if not os.path.exists(cached_data_dir):
    os.makedirs(cached_data_dir)
with open(os.path.join(cached_data_dir, "model_name.txt"), "w") as f:
    f.write(model_name)

def extract_abstract(example):
    abstract = example["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
    flg = 0 if not abstract else 1 # Excluding NULL
    return {"abstract": abstract, "NULL_flg": flg}

def process_pubmed_to_model_inputs(batch):
    inputs = tokenizer(batch["abstract"], padding="max_length", truncation=True, max_length=max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = inputs.input_ids
    batch["decoder_attention_mask"] = inputs.attention_mask
    batch["labels"] = inputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

train_data = datasets.load_dataset("pubmed", split="train[:99%]")
train_data = train_data.map(
    extract_abstract,
    remove_columns=["MedlineCitation", "PubmedData"],
    num_proc=32,
)
train_indices = [i for i, x in enumerate(train_data["NULL_flg"]) if x == 1]
train_data = train_data.select(train_indices)
train_data = train_data.map(
    process_pubmed_to_model_inputs,
    batched=True,
    batch_size=1024,
    remove_columns=["NULL_flg"],
    num_proc=32,
)

val_data = datasets.load_dataset("pubmed", split="train[-1%:]")
val_data = val_data.map(
    extract_abstract,
    remove_columns=["MedlineCitation", "PubmedData"],
    num_proc=32,
)
val_indices = [i for i, x in enumerate(val_data["NULL_flg"]) if x == 1]
val_data = val_data.select(val_indices)
val_data = val_data.map(
    process_pubmed_to_model_inputs,
    batched=True,
    batch_size=1024,
    remove_columns=["NULL_flg"],
    num_proc=32,
)

train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

# Processing for Permutation Language Modeling
import random
import numpy as np
random.seed(42)
replace_probability = 0.30

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

train_data = train_data.map(
    process_permutation_language_model_inputs,
    batched=True,
    batch_size=1024,
    num_proc=32,
)
val_data = val_data.map(
    process_permutation_language_model_inputs,
    batched=True,
    batch_size=1024,
    num_proc=32,
)


train_data.save_to_disk(os.path.join(cached_data_dir, "train"))
val_data.save_to_disk(os.path.join(cached_data_dir, "valid"))
