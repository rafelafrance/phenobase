.PHONY: test clean
.ONESHELL:

test:
	uv run -m unittest discover

clean:
	rm -rf .venv
	rm -rf build
	rm -rf scales.egg-info
	find -iname "*.pyc" -delete
