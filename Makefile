PYTHON ?= python3

.PHONY: help setup data profile blocking matching mechanism decision errors comparison test demo determinism clean

help:
	@echo "make setup        install dependencies"
	@echo "make demo         run everything end to end (about three minutes)"
	@echo "make test         run the test suite (no network, no downloaded data needed)"
	@echo "make determinism  run the pipeline twice and check the artifacts are identical"
	@echo ""
	@echo "individual stages: data profile blocking matching mechanism decision errors comparison"

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.er.fetch

profile: data
	$(PYTHON) -m src.er.profile

blocking: data
	$(PYTHON) scripts/run_blocking.py

matching: data
	$(PYTHON) scripts/run_matching.py

mechanism: data
	$(PYTHON) scripts/run_mechanism.py

decision: data
	$(PYTHON) scripts/run_decision.py

errors: data
	$(PYTHON) scripts/run_error_analysis.py

comparison: decision
	$(PYTHON) scripts/run_comparison.py

test:
	$(PYTHON) -m pytest

demo: profile blocking matching mechanism decision errors comparison
	@echo ""
	@echo "Done. Reports are in reports/, numeric artifacts in artifacts/."

determinism:
	@rm -rf .determinism && mkdir -p .determinism
	@$(MAKE) --no-print-directory blocking decision comparison >/dev/null
	@cp artifacts/*.csv .determinism/
	@$(MAKE) --no-print-directory blocking decision comparison >/dev/null
	@for f in .determinism/*.csv; do \
		cmp -s "$$f" "artifacts/$$(basename $$f)" || { echo "DIFFERS: $$(basename $$f)"; exit 1; }; \
	done
	@rm -rf .determinism
	@echo "Two consecutive runs produced identical artifacts."

clean:
	rm -rf artifacts/*.csv reports/*.md reports/*.json .determinism .pytest_cache
