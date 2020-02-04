.DELETE_ON_ERROR:

.PHONY: clean train test cli
.SECONDARY: data/processed/atis.sentences.train.csv data/processed/atis.sentences.test.csv

clean:
	rm -f data/interim/* data/processed/*

train: data/interim/atis.train.pkl
	python3 -m src.train

test: data/interim/atis.test.pkl
	python3 -m src.test

cli: data/interim/atis.test.pkl
	python3 -m src.cli

data/interim/atis.%.pkl: data/processed/atis.sentences.%.csv
	python3 -m src.data.make_dset --type $(notdir $*)

data/processed/%.csv:
	python3 src/data/conv_pkl_to_parallel.py
