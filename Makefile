.PHONY: install lint format test web-test check

install:
	pip install -r requirements-dev.txt

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

test:
	pytest -q --cov=ai_share_of_voice --cov-report=term-missing

# The web app's own tests (report sanitizer). Skipped when its packages are not installed.
web-test:
	@if [ -d web/node_modules ]; then cd web && npm test; else echo "web-test skipped: run npm ci in web/"; fi

check: lint test web-test
