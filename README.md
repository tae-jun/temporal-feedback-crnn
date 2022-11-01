# Temporal Feedback CRNN

> Code for Taejun Kim and Juhan Nam, "Temporal Feedback Convolutional Recurrent Neural Networks
> for Speech Command Recognition," APSIPA ASC, 2022 [[pdf]](https://arxiv.org/abs/1911.01803)

This repository is tested under Python 3.10.

## Preparing the dataset

```shell
curl -O http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir dataset
tar zxvf speech_commands_v0.02.tar.gz -C ./dataset
```

## Installing this package

Install [PyTorch](https://pytorch.org/) according to your environment at the official website, and run:

```shell
pip install -e .
```

It will install `tfcrnn` package and its dependencies.

## Training a model

[Weights & Biases (W&B)](https://wandb.ai/) is integrated so you can use its nice visualizations if you sign up
and log in to W&B using `wandb login`. Though, you can also run the code without an account.

By default, it will train a TF-CRNN with the basic block:

```shell
python tfcrnn/train.py
```

If you want to train another type of network, use `--skeleton cnn|crnn|tfcrnn` and `--block basic|se|resse`:

```shell
# An example for training a SampleCNN with Res-SE blocks.
python tfcrnn/train.py --skeleton cnn --block resse
```

## Citing

```
@inproceedings{taejun2022tfcrnn,
  title={Temporal Feedback Convolutional Recurrent Neural Networks for Speech Command Recognition},
  author={Kim, Taejun and Nam, Juhan},
  booktitle={Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)},
  year={2022},
  organization={IEEE}
}
```