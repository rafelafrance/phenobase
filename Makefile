.PHONY: test install dev venv clean
.ONESHELL:

VENV=.venv
PY_VER=python3.11
PYTHON=./$(VENV)/bin/$(PY_VER)
PIP_INSTALL=$(PYTHON) -m pip install

test:
	$(PYTHON) -m unittest discover

install: venv
	$(PIP_INSTALL) -U pip setuptools wheel
	$(PIP_INSTALL) .

dev: venv
	$(PIP_INSTALL) -U pip setuptools wheel
	$(PIP_INSTALL) -e .[dev]

venv:
	test -d $(VENV) || $(PY_VER) -m venv $(VENV)
	source $(VENV)/bin/activate
	$(PYTHON) -m pip install -U pip setuptools wheel

clean:
	rm -r $(VENV)
	find -iname "*.pyc" -delete
