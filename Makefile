.PHONY: *
.DEFAULT_GOAL := default

default: init test lint build

init:
	@pip install --upgrade pip
	@pip install -r requirements-dev.txt
	@pip install -r requirements.txt

test:
	@pytest -v --cov=l2m2 --cov=test_utils --cov-report=term-missing --failed-first --durations=0

coverage:
	@pytest --cov=l2m2 --cov=test_utils --cov-report=html
	@open htmlcov/index.html

lint:
	@flake8 .
	@mypy .

build:
	@python -m build

clean:
	@rm -rf build \
		dist \
		*.egg-info \
		.pytest_cache \
		.mypy_cache \
		htmlcov \
		.coverage
	@find . -type d -name __pycache__ -exec rm -r {} +

publish: clean build
	@twine upload dist/*

update-models:
	@python scripts/create_model_table.py | pbcopy