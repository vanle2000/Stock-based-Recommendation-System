
.PHONY: install test lint clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src/ tests/ --max-line-length=100

clean:
	find . -type f -name "*.pyc" -delete
