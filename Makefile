.PHONY: test install dev venv clean
.ONESHELL:

test:
	python3 -m unittest discover

install: venv
	. .venv/bin/activate
	python3 -m pip install -U pip setuptools wheel
	python3 -m pip install .

dev: venv
	. .venv/bin/activate
	python3 -m pip install -U pip setuptools wheel
	python3 -m pip install -e .[dev]
	pre-commit install

venv:
	test -d .venv || python3.12 -m venv .venv
	. .venv/bin/activate
	python3 -m pip install -U pip setuptools wheel

clean:
	rm -r .venv
	find -iname "*.pyc" -delete
