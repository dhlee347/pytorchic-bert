[<img width="400"
src="https://user-images.githubusercontent.com/32828768/49876264-ff2e4180-fdf0-11e8-9512-06ffe3ede9c5.png">](https://jalammar.github.io/illustrated-bert/)

# Pytorchic BERT
This is re-implementation of [Google BERT model](https://github.com/google-research/bert) [[paper](https://arxiv.org/abs/1810.04805)] in Pytorch. I was strongly inspired by [Hugging Face's code](https://github.com/huggingface/pytorch-pretrained-BERT) and I referred a lot to their codes, but I tried to make my codes **more pythonic and pytorchic style**. Actually, the number of lines is less than a half of HF's. 

(It is still not so heavily tested - let me know when you find some bugs.)

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
Download pretrained model [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) and
[GLUE Benchmark Datasets]( https://github.com/nyu-mll/GLUE-baselines) 
before fine-tuning.
* make sure that "total_steps" in train_mrpc.json is n_epochs*(num_data/batch_size)
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
[Google BERT original repo](https://github.com/google-research/bert) also reported 84.5%.


### Pre-training Transformer
Input file format :
1. One sentence per line. These should ideally be actual sentences, not entire paragraphs or arbitrary spans of text. (Because we use the sentence boundaries for the "next sentence prediction" task).
2. Blank lines between documents. Document boundaries are needed so that the "next sentence prediction" task doesn't span between documents.
```
Document 1 sentence 1
Document 1 sentence 2
...
Document 1 sentence 45

Document 2 sentence 1
Document 2 sentence 2
...
Document 2 sentence 24
```
Usage :
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
Output (with Toronto Book Corpus):
```
cuda (8 GPUs)
Iter (loss=5.837): : 30089it [18:09:54,  2.17s/it]
Epoch 1/25 : Average Loss 13.928
Iter (loss=3.276): : 30091it [18:13:48,  2.18s/it]
Epoch 2/25 : Average Loss 5.549
Iter (loss=4.163): : 7380it [4:29:38,  2.19s/it]
...
```
Training Curve (1 epoch ~ 30k steps ~ 18 hours):

Loss for Masked LM vs Iteration steps
<img src="https://user-images.githubusercontent.com/32828768/50011629-9a0e5380-ff8a-11e8-87ab-18cd22453561.png">
Loss for Next Sentence Prediction vs Iteration steps
<img src="https://user-images.githubusercontent.com/32828768/50011633-9c70ad80-ff8a-11e8-8670-8baaebb6e51a.png">

