export PYTHONPATH := src

install:
	pip install -r requirements.txt

data:
	python -m dvf.download

train:
	python -m dvf.train

api:
	uvicorn api.main:app --reload

test:
	pytest -v

.PHONY: install data train api test
