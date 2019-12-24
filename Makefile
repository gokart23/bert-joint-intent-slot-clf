.DELETE_ON_ERROR:

.PHONY: clean train
.SECONDARY: data/processed/atis.sentences.train.csv data/processed/atis.sentences.test.csv

clean:
	rm -f data/interim/* data/processed/*

train: data/interim/atis.train.pkl

data/interim/atis.%.pkl: data/processed/atis.sentences.%.csv
	python3 -m src.data.make_dset --type $(notdir $*)

data/processed/%.csv:
	python3 src/data/conv_pkl_to_parallel.py
