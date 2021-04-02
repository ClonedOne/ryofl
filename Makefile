init:
	pip install -r requirements.txt

test:
	pytest tests

install:
	pip install --editable .


.PHONY: init install test

