venv:
	python3 -m venv .venv

setup: venv
	pip install --upgrade pip
	pip install -r requirements.txt
	python -c "import torch, streamlit, transformers, timm; print('All packages imported successfully!')"

run:
	source .venv/bin/activate; streamlit run app.py --server.port $PORT --server.headless true


