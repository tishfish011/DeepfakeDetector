setup:
	python3 -m venv .venv; \
	pip3 install --upgrade pip; \
	source .venv/bin/activate; \
		pip3 install -r requirements.txt; \
		python -c "import torch, streamlit, transformers, timm; print('All packages imported successfully!')"

run:
	source .venv/bin/activate; streamlit run app.py --server.port 9999 --server.headless true

stop:
	deactivate
