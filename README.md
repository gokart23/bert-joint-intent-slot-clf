Dataset
=======

A pickled form of raw ATIS data can be downloaded from [here](https://github.com/howl-anderson/ATIS_dataset/raw/master/data/raw_data/ms-cntk-atis/atis.train.pkl). This raw data can be converted to parallel data via `scripts/conv_pkl_to_parallel.py`.

Train and Inference
===================

In order to train a model, run `make train`.
In order to run inference on the test set, run `make test`.

In order to start afresh and remove all intermediate files, run `make clean`.
