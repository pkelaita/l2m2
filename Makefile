.PHONY: *
.DEFAULT_GOAL := default

VERSION := $(shell python -c "from l2m2 import __version__; print(__version__)")

default: lint typecheck test

init:
	pip install --upgrade pip
	pip install -r requirements-dev.txt

test:
	pytest -v --cov=l2m2 --cov=test_utils --cov-report=term-missing --failed-first --durations=0

itest:
	@pip install dist/l2m2-$(VERSION)-py3-none-any.whl > /dev/null
	python integration_tests/itests.py
	@pip uninstall -y l2m2 > /dev/null

coverage:
	pytest --cov=l2m2 --cov=test_utils --cov-report=html
	open htmlcov/index.html

lint:
	-flake8 .

typecheck:
	-mypy .

build:
	python -m build

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
	twine upload dist/*

update-readme:
	cd scripts && ./update_badges.sh
	cd scripts && python3 update_model_table.py