PYTHON ?= .venv/bin/python
OUTPUT_DIR ?=

.PHONY: help check-data run-comparison analyze

help:
	@printf '%s\n' "Targets:"
	@printf '%s\n' "  check-data      Inspect the RGB-D sequence and synchronization."
	@printf '%s\n' "  run-comparison  Run the full ORB vs SuperPoint + LightGlue comparison."
	@printf '%s\n' "  analyze         Analyze a comparison run and regenerate plots."
	@printf '%s\n' ""
	@printf '%s\n' "Variables:"
	@printf '%s\n' "  PYTHON=.venv/bin/python"
	@printf '%s\n' "  OUTPUT_DIR=outputs/comparison_YYYY-MM-DD_HH-MM-SS"

check-data:
	$(PYTHON) scripts/prepare_rgbd_sequence.py

run-comparison:
	$(PYTHON) scripts/run_comparison.py

analyze:
	$(PYTHON) scripts/analyze_comparison.py $(OUTPUT_DIR)
