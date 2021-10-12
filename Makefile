SHELL:=/bin/bash

all:
	@echo "choose explicit target = type 'make ' and press TAB"

S=codes
I=data
O=out


# ===== MAIN STUFF 

SCRIPT=$S/bert.py
do:
	python3 $(SCRIPT) --input test --divisor 3 --first-word-pooling > featuresfinal_test.txt
	python3 $(SCRIPT) --input investigate --divisor 3 --all-word-pooling --verbose > featuresfinal_investigate.txt

