# RNSR Makefile
# Run `make help` to see available commands

# Use virtual environment if it exists, otherwise system python
VENV := .venv
ifeq ($(wildcard $(VENV)/bin/python),)
    PYTHON := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)
else
    PYTHON := $(VENV)/bin/python
endif

.PHONY: help demo test test-fast test-cov lint format install clean venv update switch benchmark-timeline benchmark-contradiction benchmark-features benchmark-financebench

# Default target
help:
	@echo "RNSR - Recursive Neural-Symbolic Retriever"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Demo & Benchmarks:"
	@echo "  demo              Launch the Gradio web demo (http://localhost:7860)"
	@echo "  demo-office       Run presentation-ready office demo (local only)"
	@echo "  benchmark         Run performance benchmarks"
	@echo "  benchmark-compare Compare RNSR vs baselines (small doc)"
	@echo "  benchmark-large   Compare on LARGE doc (shows RNSR advantage)"
	@echo "  benchmark-timeline     Run timeline benchmark (LexTime + synthetic)"
	@echo "  benchmark-contradiction Run contradiction detection benchmark"
	@echo "  benchmark-features     Run both timeline + contradiction benchmarks"
	@echo "  benchmark-financebench Run FinanceBench subset (15 Qs, ~3-4 hrs)"
	@echo ""
	@echo "Testing:"
	@echo "  test          Run all tests"
	@echo "  test-fast     Run tests without slow integration tests"
	@echo "  test-cov      Run tests with coverage report"
	@echo ""
	@echo "Development:"
	@echo "  update        Pull the latest version from git"
	@echo "  switch        Switch git branch (interactive numbered list)"
	@echo "  lint          Run linter (ruff)"
	@echo "  format        Format code (ruff)"
	@echo "  venv          Create virtual environment"
	@echo "  install       Install dependencies"
	@echo "  install-dev   Install dev dependencies"
	@echo "  clean         Clean build artifacts"
	@echo ""
	@echo "Using Python: $(PYTHON)"
	@echo ""

# Create virtual environment
venv:
	@echo "🐍 Creating virtual environment..."
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	@echo ""
	@echo "✅ Virtual environment created at $(VENV)"
	@echo "   Run 'make install-dev' to install dependencies"

# Launch the Gradio web demo
demo:
	@echo "🚀 Starting RNSR Demo..."
	@echo "   Open http://localhost:7860 in your browser"
	@echo ""
	$(PYTHON) demo.py

# Run presentation-ready office demo
demo-office:
	@echo "🎬 Starting Office Demo..."
	@echo ""
	$(PYTHON) scripts/office_demo.py

# Run office demo with a specific PDF
demo-pdf:
	@echo "Usage: make demo-pdf PDF=path/to/document.pdf"
	@test -n "$(PDF)" || (echo "Error: PDF is required" && exit 1)
	$(PYTHON) scripts/office_demo.py --pdf $(PDF)

# Run benchmarks
benchmark:
	@echo "📊 Running Benchmarks..."
	@echo ""
	$(PYTHON) scripts/benchmark_demo.py

# Run benchmark with specific PDF
benchmark-pdf:
	@echo "Usage: make benchmark-pdf PDF=path/to/document.pdf"
	@test -n "$(PDF)" || (echo "Error: PDF is required" && exit 1)
	$(PYTHON) scripts/benchmark_demo.py --pdf $(PDF)

# Compare RNSR vs Naive RAG vs Long Context (small document)
benchmark-compare:
	@echo "📊 Running Comparison Benchmark..."
	@echo "   Comparing: RNSR vs Naive RAG vs Long Context LLM"
	@echo ""
	$(PYTHON) scripts/compare_benchmarks.py --quick

# Compare on LARGE document (shows RNSR advantage when Long Context truncates)
benchmark-large:
	@echo "📊 Running LARGE Document Benchmark..."
	@echo "   This demonstrates RNSR's advantage on documents too large for context"
	@echo ""
	$(PYTHON) scripts/compare_benchmarks.py --large

# Run timeline benchmark (LexTime + synthetic PDFs)
benchmark-timeline:
	@echo "📊 Running Timeline Benchmark..."
	@echo "   Tier 1: Synthetic PDF ground truth"
	@echo "   Tier 2: LexTime temporal ordering (514 instances)"
	@echo ""
	$(PYTHON) -c "from rnsr.benchmarks.timeline_bench import run_timeline_benchmark; import json; r = run_timeline_benchmark(); print(json.dumps(r, indent=2, default=str))"

# Run contradiction detection benchmark
benchmark-contradiction:
	@echo "📊 Running Contradiction Benchmark..."
	@echo "   Single-doc: Greenfield Annual Report (5 known contradictions)"
	@echo "   Cross-doc: Expert Reports A/B + Incident Report (6 known contradictions)"
	@echo ""
	$(PYTHON) -c "from rnsr.benchmarks.contradiction_bench import run_contradiction_benchmark; import json; r = run_contradiction_benchmark(); print(json.dumps(r, indent=2, default=str))"

# Run both feature benchmarks (timeline + contradiction)
benchmark-features:
	@echo "📊 Running Feature Benchmarks (Timeline + Contradiction)..."
	@echo ""
	@$(MAKE) benchmark-timeline
	@echo ""
	@$(MAKE) benchmark-contradiction

# Run FinanceBench subset (15 questions across 15 SEC filings — takes ~3-4 hours)
benchmark-financebench:
	@echo "📊 Running FinanceBench Subset (15 questions, 15 SEC filings)..."
	@echo "   This downloads 10-K/10-Q PDFs, ingests, builds KG, and answers."
	@echo "   Expect ~15-20 min per question. Results saved to benchmark_results/"
	@echo ""
	$(PYTHON) run_financebench_subset.py

# Compare with your own PDF
benchmark-compare-pdf:
	@echo "Usage: make benchmark-compare-pdf PDF=path/to/document.pdf"
	@test -n "$(PDF)" || (echo "Error: PDF is required" && exit 1)
	$(PYTHON) scripts/compare_benchmarks.py --pdf $(PDF)

# Run all tests
test:
	@echo "🧪 Running all tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short

# Run tests without slow tests (e.g., e2e that need LLM)
test-fast:
	@echo "🧪 Running fast tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short -m "not slow" --ignore=tests/test_e2e_workflow.py

# Run tests with coverage
test-cov:
	@echo "🧪 Running tests with coverage..."
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=rnsr --cov-report=html --cov-report=term-missing
	@echo ""
	@echo "📊 Coverage report: htmlcov/index.html"

# Run specific test file
test-file:
	@echo "Usage: make test-file FILE=tests/test_xyz.py"
	@test -n "$(FILE)" || (echo "Error: FILE is required" && exit 1)
	$(PYTHON) -m pytest $(FILE) -v --tb=short

# Lint code
lint:
	@echo "🔍 Linting code..."
	$(PYTHON) -m ruff check rnsr/ tests/ demo.py

# Format code
format:
	@echo "✨ Formatting code..."
	$(PYTHON) -m ruff format rnsr/ tests/ demo.py
	$(PYTHON) -m ruff check --fix rnsr/ tests/ demo.py

# Type check
typecheck:
	@echo "🔎 Type checking..."
	$(PYTHON) -m mypy rnsr/ --ignore-missing-imports

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	$(PYTHON) -m pip install -e .

# Install dev dependencies
install-dev:
	@echo "📦 Installing dev dependencies..."
	$(PYTHON) -m pip install -e ".[dev,openai,anthropic,gemini,demo]"

# Install all dependencies including benchmarks
install-all:
	@echo "📦 Installing all dependencies..."
	$(PYTHON) -m pip install -e ".[all,dev,benchmarks,demo]"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Done!"

# Build package
build:
	@echo "📦 Building package..."
	$(PYTHON) -m build

# Publish to PyPI (requires twine)
publish:
	@echo "📤 Publishing to PyPI..."
	@echo "Make sure you have a PyPI account and API token configured."
	twine upload dist/*

# Publish to TestPyPI first
publish-test:
	@echo "📤 Publishing to TestPyPI..."
	twine upload --repository testpypi dist/*

# Pull the latest version from git
update:
	@echo "⬇️  Pulling latest changes..."
	git pull
	@echo "✅ Up to date."

# Interactive branch switcher
switch:
	@echo "🔀 Available branches:"
	@echo ""
	@git fetch --prune --quiet 2>/dev/null || true
	@branches=$$(git branch -a --sort=-committerdate \
		| sed 's/^[* ]*//' \
		| sed 's|remotes/origin/||' \
		| grep -v '^HEAD ' \
		| awk '!seen[$$0]++' \
		| head -10); \
	i=1; \
	for b in $$branches; do \
		current=""; \
		if git branch --show-current 2>/dev/null | grep -qx "$$b"; then \
			current=" (current)"; \
		fi; \
		echo "  $$i) $$b$$current"; \
		i=$$((i + 1)); \
	done; \
	echo ""; \
	printf "Enter branch number (1-10): "; \
	read choice; \
	target=$$(echo "$$branches" | sed -n "$${choice}p"); \
	if [ -z "$$target" ]; then \
		echo "❌ Invalid selection."; \
		exit 1; \
	fi; \
	echo ""; \
	echo "Switching to: $$target"; \
	if git show-ref --verify --quiet "refs/heads/$$target" 2>/dev/null; then \
		git checkout "$$target"; \
	else \
		git checkout -b "$$target" "origin/$$target" 2>/dev/null \
			|| git checkout "$$target"; \
	fi; \
	echo "✅ Now on branch: $$(git branch --show-current)"

# Check if environment is set up correctly
check-env:
	@echo "🔍 Checking environment..."
	@echo ""
	@echo "Python path: $(PYTHON)"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PYTHON) -m pip --version)"
	@echo ""
	@echo "API Keys:"
	@test -n "$$GOOGLE_API_KEY" && echo "  ✅ GOOGLE_API_KEY is set" || echo "  ❌ GOOGLE_API_KEY not set"
	@test -n "$$OPENAI_API_KEY" && echo "  ✅ OPENAI_API_KEY is set" || echo "  ❌ OPENAI_API_KEY not set"
	@test -n "$$ANTHROPIC_API_KEY" && echo "  ✅ ANTHROPIC_API_KEY is set" || echo "  ❌ ANTHROPIC_API_KEY not set"
	@echo ""
