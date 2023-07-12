# BioNART
The implementation of "[BioNART: A Biomedical Non-AutoRegressive Transformer for Natural Language Generation](https://aclanthology.org/2023.bionlp-1.34/)"
![image](https://github.com/aistairc/BioNART/assets/20109895/68d177bb-c825-41e9-b5d7-fdb7e4381211)

This implementation is intended to run on the [ABCI](https://abci.ai/)


## Requirements
```
pip install -r requirements.txt
```

## Pre-training on PubMed
```
cd preprocess_caches
qsub -g GROUPNAME run.sh
cd ..
qsub -g GROUPNAME job_pretraining.sh
```

## Fine-tuning on Biomedical Text Generation tasks
```
qsub -g GROUPNAME job_bio_generation.sh
```

## Acknowledgements
This work is based on results obtained from a project JPNP20006, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).

