.PHONY: install data eda train test lint api docker-build docker-run
install:
	python -m pip install -r requirements.txt
data:
	python -m src.data
eda:
	python -m src.eda
train:
	python -m src.train
test:
	pytest -q
lint:
	ruff check src tests
api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000
docker-build:
	docker build -t heart-disease-mlops:local .
docker-run:
	docker run --rm -p 8000:8000 heart-disease-mlops:local
