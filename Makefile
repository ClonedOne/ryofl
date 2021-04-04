.PHONY: init install test

init:
	pip install -r requirements.txt

test:
	pytest tests

install:
	pip install --editable .



