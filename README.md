# Pytorchic BERT

This is Pytorch re-implementation of [Google BERT model](https://github.com/google-research/bert) [[paper](https://arxiv.org/abs/1810.04805)]. I'm strongly inspired by [Hugging Face's code](https://github.com/huggingface/pytorch-pretrained-BERT) and I referred a lot to their codes, but I tried to make my codes more pythonic and pytorchic style. Actually, the number of lines is less than a half of HF's.

## Requirement

Python > 3.6
fire
tqdm
tensorflow (for loading checkpoint file)
tensorboardx

## Overview

This contains 9 python files.
- [`tokenization.py`](./tokenization.py) : Tokenizers adopted from the original Google BERT's code
- [`checkpoint.py`](./checkpoint.py) : Functions to load a model from tensorflow's checkpoint file
- [`models.py`](./models.py) : Model classes for a general transformer
- [`optim.py`](./optim.py) : A custom optimizer (BertAdam class) adopted from Hugging Face's code
- [`train.py`](./train.py) : A helper class for training and evaluation
- [`utils.py`](./utils.py) : Several utility functions
- [`pretrain.py`](./pretrain.py) : An example code for pre-training transformer
- [`classify.py`](./classify.py) : An example code for fine-tuning using pre-trained transformer

## Example Usage

#### Fine-tuning (MRPC) Classifier with Pre-trained Transformer
```
export GLUE_DIR=/path/to/glue
export BERT_PRETRAIN=/path/to/pretrain
export SAVE_DIR=/path/to/save

python classify.py \
    --task mrpc \
    --mode train \
    --train_cfg config/train_mrpc.json \
    --model_cfg config/bert_base.json \
    --data_file $GLUE_DIR/MRPC/train.tsv \
    --pretrain_file $BERT_PRETRAIN/bert_model.ckpt \
    --vocab $BERT_PRETRAIN/vocab.txt \
    --save_dir $SAVE_DIR \
    --max_len 128
```
Output :
```
cuda (8 GPUs)
Iter (loss=0.465): 100%|█████████████████████████| 115/115 [01:22<00:00,  2.04it/s]
Epoch 1/15 : Average Loss 0.601
Iter (loss=0.618): 100%|█████████████████████████| 115/115 [00:52<00:00,  2.26it/s]
Epoch 2/15 : Average Loss 0.437
Iter (loss=0.051): 100%|█████████████████████████| 115/115 [00:52<00:00,  2.32it/s]
Epoch 3/15 : Average Loss 0.245
                                    ...
Iter (loss=0.000): 100%|█████████████████████████| 115/115 [00:52<00:00,  2.34it/s]
Epoch 15/15 : Average Loss 0.007
```

#### Evaluation of trained Classifier
```
export GLUE_DIR=/path/to/glue
export BERT_PRETRAIN=/path/to/pretrain
export SAVE_DIR=/path/to/save

python classify.py \
    --task mrpc \
    --mode eval \
    --train_cfg config/train_mrpc.json \
    --model_cfg config/bert_base.json \
    --data_file $GLUE_DIR/MRPC/dev.tsv \
    --model_file $SAVE_DIR/model_epoch_15_steps_1700.pt \
    --vocab $BERT_PRETRAIN/vocab.txt \
    --max_len 128
```
Output :
```
cuda (8 GPUs)
Iter(acc=0.792): 100%|███████████████████████████| 13/13 [00:27<00:00,  2.02it/s]
Accuracy: 0.8308823704719543
```

You should see 83%~85% accuracy in MRPC task


#### Pre-train Transformer
```
export DATA_FILE=/path/to/tbc/books_large_all.txt
export BERT_PRETRAIN=/path/to/pretrain
export SAVE_DIR=/path/to/save

python pretrain.py \
    --train_cfg config/pretrain.json \
    --model_cfg config/bert_base.json \
    --data_file $DATA_FILE \
    --vocab $BERT_PRETRAIN/vocab.txt \
    --save_dir $SAVE_DIR \
    --max_len 512 \
    --max_pred 20 \
    --mask_prob 0.15
```
Output :
```
cuda (8 GPUs)
Epoch(loss=3.468):  12%|███▊                            | 3/25 [54:16:45<398:02:53, 65135.16s/it]
Iter (loss=3.035): : 25409it [15:21:14,  2.18s/it]
```