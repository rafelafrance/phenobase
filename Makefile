.PHONY: test install dev base clean
.ONESHELL:

test:
	python3 -m unittest discover

base:
	test -d .venv || python3.12 -m venv .venv
	. .venv/bin/activate
	python3 -m pip install -U pip setuptools wheel

install: base
	python3 -m pip install .

dev: base
	python3 -m pip install -e .[dev]
	pre-commit install

clean:
	rm -r .venv
	find -iname "*.pyc" -delete
