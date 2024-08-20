
The source codes are based on https://github.com/westlake-repl/Adapter4Rec.

# Dataset

The complete recommendation dataset (MovieLens) is available under the Dataset directory.
Datasets are preprocessed from the MovieLens dataset with ./Dataset/movielens/movie_lens_dataset.ipynb

It was run on Google Colab.

# Experiements

The experiments are run on Eddie cluster of UoE. Job submission are done via ./Downstream/Text/script/rec.sh and ./Pretraining/Text/script/rec.sh

## Requirements of Python packages

- pytorch == 2.3.1+cuda12.1
- transformers==4.42.4
- loralib==0.1.2

## Pre-trained Model Download

The pre-trained Item Modality Encoder is the BERT model from https://huggingface.co/bert-base-uncased.

The end-to-end pretrained checkpoint used in transfer learning (experiments under ./Downstream) are from the paper by Fu et al. (https://arxiv.org/pdf/2305.15036), and can be downloaded from https://drive.google.com/file/d/16xIo2ygB4b3ERrg81zDzBXXUfdLw86Ss/view. This SASRec+BERT pretrained checkpoint should be placed under ./Downstream/Text/pretrained_models before starting experiments.

# Training

cd Downstream/Text/script
qsub rec.sh (on Eddie cluster)
