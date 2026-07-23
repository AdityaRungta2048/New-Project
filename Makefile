.PHONY: help install test demo api ui docker clean

help:
	@echo "make install   - install dependencies"
	@echo "make test      - run the offline test suite"
	@echo "make demo      - run the four canonical cases (add WRITE=1 to save docs)"
	@echo "make api       - launch the FastAPI service on :8000"
	@echo "make ui        - launch the Streamlit Verdict Explorer on :8501"
	@echo "make docker    - build and run the full stack (API + UI + Ollama)"
	@echo "make clean     - remove caches and the local audit store"

install:
	pip install -r requirements.txt

test:
	pytest

demo:
	python -m examples.demo $(if $(WRITE),--write,)

api:
	uvicorn arbiter.api:app --reload --port 8000

ui:
	streamlit run ui/streamlit_app.py

docker:
	docker compose up --build

clean:
	rm -rf data __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
