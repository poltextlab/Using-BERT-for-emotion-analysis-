SHELL:=/bin/bash

all:
	@echo "choose explicit target = type 'make ' and press TAB"

S=codes
I=data
O=out


# ===== MAIN STUFF 

SCRIPT=$S/bert.py
do: test investigate

test:
	python3 $(SCRIPT) --input test --divisor 3 --first-word-pooling > featuresfinal_test.txt

investigate:
	python3 $(SCRIPT) --input investigate --divisor 3 --all-word-pooling --verbose > featuresfinal_investigate.txt

# an example run :)
# should be parametrized: e.g. '-i' in gridsearch_BERT.py ...
POOLING=--mean-pooling
example:
	python3 $(SCRIPT) --input etl_nothree --divisor 200 $(POOLING) > featuresfinal_etl_nothree.txt
	python3 codes/gridsearch_BERT.py
	cat clasrep_bert_etl_nothree.json | python3 -m json.tool | grep -A 3 "weighted avg"

