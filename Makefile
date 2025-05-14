.PHONY: *
.DEFAULT_GOAL := default

VERSION := $(shell uv run python -c "from l2m2 import __version__; print(__version__)")

default: lint type test

init:
	uv sync

test:
	uv run pytest -v --cov=l2m2 --cov-report=term-missing --failed-first --durations=0

tox:
	uv run tox -p auto

clear-deps:
	@uv pip uninstall l2m2 > /dev/null 2>&1
	@uv pip freeze | xargs uv pip uninstall > /dev/null

itest-run:
	@uv pip install dist/l2m2-$(VERSION)-py3-none-any.whl > /dev/null
	@uv pip install python-dotenv > /dev/null
	@uv run tests/integration/itests.py

itest: clear-deps itest-run clear-deps

itl:
	@uv run tests/integration/itests.py --local

coverage:
	uv run pytest --cov=l2m2 --cov-report=html
	open htmlcov/index.html

lint:
	-uv run ruff check .

type:
	-uv run ty check l2m2

type-mypy:
	-uv run mypy .

build:
	uv build

clean:
	@rm -rf build \
		dist \
		*.egg-info \
		.pytest_cache \
		.mypy_cache \
		htmlcov \
		.coverage \
		*.lcov
	@find . -type d -name __pycache__ -exec rm -r {} +

publish: clean build
	uv run twine upload dist/*

update-docs:
	cd scripts && ./update_badges.sh
	cd scripts && python3 update_models.py