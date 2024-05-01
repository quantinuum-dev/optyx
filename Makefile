.PHONY: format lint test coverage docs

SRC ?= optyx
DOCS ?= docs

format:
	black $(SRC)

# Just prints the diff of the changes we formatter wants to make
format-diff:
	black --diff $(SRC)

lint:
	pflake8 $(SRC) & pylint $(SRC)

test:
	coverage run -m pytest $(SRC)

coverage: test
	 coverage report --fail-under=95 --show-missing

docs:
	sphinx-build $(DOCS) $(DOCS)/_build/html
