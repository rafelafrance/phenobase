.PHONY: test install dev venv clean
.ONESHELL:

VENV=.venv
PYTHON=./$(VENV)/bin/python3.10

test:
	$(PYTHON) -m unittest discover

install: venv
	source $(VENV)/bin/activate
	$(PYTHON) -m pip install -U pip setuptools wheel
	$(PYTHON) -m pip install .

dev: venv
	source $(VENV)/bin/activate
	$(PYTHON) -m pip install -U pip setuptools wheel
	$(PYTHON) -m pip install -e .[dev]
	pre-commit install

venv:
	test -d $(VENV) || python3.10 -m venv $(VENV)

clean:
	rm -r $(VENV)
	find -iname "*.pyc" -delete
