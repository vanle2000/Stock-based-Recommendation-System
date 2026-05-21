
.PHONY: install install-api test lint serve docker-build docker-run eval clean

install:
	pip install -r requirements.txt

install-api:
	pip install -r requirements_api.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src/ tests/ --max-line-length=100

# Run the FastAPI service locally
serve:
	uvicorn src.api.app:app --reload --port 8000

# Build Docker image
docker-build:
	docker build -t stock-recommender:latest .

# Run containerised service
docker-run:
	docker run -p 8000:8000 --rm stock-recommender:latest

# Run offline recommendation evaluation
eval:
	python -m src.models.recommender_eval

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
