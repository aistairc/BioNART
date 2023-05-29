import os
import numpy as np
import datasets
import transformers
import evaluate

from nar_transformers import BertTokenizerFast
from nar_transformers import EncoderDecoderConfig, EncoderDecoderModel
from nar_transformers import TrainingArguments, Trainer, EvalPrediction

import wandb
os.environ["WANDB_PROJECT"] = "summarization-bio"


model_name = "dmis-lab/biobert-base-cased-v1.1"
#model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
#model_name = "allenai/scibert_scivocab_cased"
batch_size = 20  # change to 16 for full training
max_length = 128 # 128 actually works better for MT
latent_size = 8

tokenizer = BertTokenizerFast.from_pretrained(model_name)

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(batch["src"], padding="max_length", truncation=True, max_length=max_length)
    outputs = tokenizer(batch["tgt"], padding="max_length", truncation=True, max_length=max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

json_dir = "/scratch/aae15163zd/data/bio/prepared_data/covid_dialogue"
train_data = datasets.load_dataset("json", data_files=os.path.join(json_dir, "train.json"), field="data", split="train")
val_data = datasets.load_dataset("json", data_files=os.path.join(json_dir, "dev.json"), field="data", split="train")

train_data = train_data.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=1024,
    remove_columns=["src", "tgt"],
    num_proc=32, # set to the number of CPU cores in AF node
)
val_data = val_data.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=1024,
    remove_columns=["src", "tgt"],
    num_proc=32,
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name, latent_size)
#model = EncoderDecoderModel.from_pretrained(
#    "/groups/gac50543/migrated_from_SFA_GPFS/asada/pretraining/pubmed-biobert-base-cased-plm0.30-lr1e-4-bs2560/checkpoint-45000/",
#)
model.config.is_vae = False
model.config.dropout_prob = 0.5

# set special tokens
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# load bleu for validation
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

def compute_metrics(p: EvalPrediction):
    label_ids = p.label_ids
    pred_ids = p.predictions
    # Removing repetition tokens
    no_rep_pred_ids = [
        [x[i] if i == 0 or x[i-1] != x[i] else tokenizer.pad_token_id for i in range(len(x))] for x in pred_ids
    ]
    no_rep_pred_str = tokenizer.batch_decode(no_rep_pred_ids, skip_special_tokens=True)

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    #rouge_output = rouge.compute(predictions=pred_str, references=label_str)
    rouge_output = rouge.compute(predictions=no_rep_pred_str, references=label_str)
    bleu_output = bleu.compute(predictions=no_rep_pred_str, references=label_str, max_order=4)

    return {
        "rouge1": round(np.mean(rouge_output["rouge1"]), 4),
        "rouge2": round(np.mean(rouge_output["rouge2"]), 4),
        "rougeL": round(np.mean(rouge_output["rougeL"]), 4),
        "bleu4": round(np.mean(bleu_output["bleu"]), 4),
    }

run_name = "covid-1e-5-from-biobert"
# set training arguments - these params are not really tuned, feel free to change
training_args = TrainingArguments(
    output_dir=os.path.join("~/my_data/summarization-bio/", run_name),
    evaluation_strategy="steps",
    save_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    logging_steps=1,  # set to 1000 for full training
    save_steps=1,  # set to 500 for full training
    eval_steps=1,  # set to 8000 for full training
    warmup_ratio=0.1,
    learning_rate=2e-05,
    num_train_epochs=3,
    #max_steps=20_000,
    overwrite_output_dir=True,
    save_total_limit=8,
    bf16=True,
    torch_compile=False,
    weight_decay=0.1,
    report_to="wandb",
    run_name=run_name,
    gradient_accumulation_steps=1,
)

# instantiate trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)

trainer.train()
#trainer.train(resume_from_checkpoint=True)
