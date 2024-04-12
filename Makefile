.PHONY: *
.DEFAULT_GOAL := lint

init:
	@pip install --upgrade pip
	@pip install -r requirements-dev.txt
	@pip install -r requirements.txt

test:
	@pytest -v --cov=l2m2 --cov-report=term-missing --failed-first --durations=0

coverage:
	@pytest --cov=l2m2 --cov-report=html

lint:
	@flake8 l2m2 tests

build:
	@python -m build

clean:
	@rm -rf build \
		dist \
		*.egg-info \
		.pytest_cache \
		htmlcov \
		.coverage
	@find . -type d -name __pycache__ -exec rm -r {} +

publish: clean build
	@twine upload dist/*

update-models:
	@python scripts/create_model_table.py | pbcopy