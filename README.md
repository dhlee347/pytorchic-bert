[<img width="400"
src="https://user-images.githubusercontent.com/32828768/49876264-ff2e4180-fdf0-11e8-9512-06ffe3ede9c5.png">](https://jalammar.github.io/illustrated-bert/)

# Pytorchic BERT
This is re-implementation of [Google BERT model](https://github.com/google-research/bert) [[paper](https://arxiv.org/abs/1810.04805)] in Pytorch. I'm strongly inspired by [Hugging Face's code](https://github.com/huggingface/pytorch-pretrained-BERT) and I referred a lot to their codes, but I tried to make my codes more pythonic and pytorchic style. Actually, the number of lines is less than a half of HF's.

(It is still not so heavily tested - let me know when you find some bugs)

## Requirements

Python > 3.6, fire, tqdm, tensorboardx,
tensorflow (for loading checkpoint file)

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

### Fine-tuning (MRPC) Classifier with Pre-trained Transformer
Download [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) and
[GLUE Benchmark Datasets]( https://github.com/nyu-mll/GLUE-baselines) 
before fine-tuning.
* make sure that "total_steps" in train_mrpc.json should be n_epochs*(num_data/batch_size)
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
Iter (loss=0.308): 100%|██████████████████████████████████████████████| 115/115 [01:19<00:00,  2.07it/s]
Epoch 1/3 : Average Loss 0.547
Iter (loss=0.303): 100%|██████████████████████████████████████████████| 115/115 [00:50<00:00,  2.30it/s]
Epoch 2/3 : Average Loss 0.248
Iter (loss=0.044): 100%|██████████████████████████████████████████████| 115/115 [00:50<00:00,  2.33it/s]
Epoch 3/3 : Average Loss 0.068
```

### Evaluation of the trained Classifier
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
    --model_file $SAVE_DIR/model_steps_345.pt \
    --vocab $BERT_PRETRAIN/vocab.txt \
    --max_len 128
```
Output :
```
cuda (8 GPUs)
Iter(acc=0.792): 100%|████████████████████████████████████████████████| 13/13 [00:27<00:00,  2.01it/s]
Accuracy: 0.843137264251709
```
You should see 84%~88% accuracy in this task


### Pre-training Transformer
```
export DATA_FILE=/path/to/corpus
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
Iter (loss=2.805): : 25584it [15:27:36,  2.18s/it]
```
Training Curve, loss vs steps (only for masked LM) :
<img src="https://user-images.githubusercontent.com/32828768/49846589-e47cae00-fd99-11e8-9c19-a29e832bf480.png">
pretraining with Toronto Book Corpus, 100k steps ~ 2.5 days