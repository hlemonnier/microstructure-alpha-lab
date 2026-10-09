PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: help e2e check-repository verify-results verify-evidence-gates external-readiness test laptop-smoke cloud-package modal-study kelly-candidate
.DEFAULT_GOAL := help

help:
	@printf '%s\n' 'make e2e                  Build, install and verify synthetic E2E artifacts' 'make check-repository     Check public files, links and credential patterns' 'make test                 Run the existing implementation regressions' 'make cloud-package        Export the clean source project' 'make verify-evidence-gates Audit empirical evidence (may remain incomplete)'

e2e:
	$(PYTHON) scripts/run_public_e2e.py

check-repository:
	$(PYTHON) scripts/check_repository.py

verify-results:
	bash scripts/verify_results.sh

verify-evidence-gates:
	bash scripts/verify_remaining_evidence_gates.sh

external-readiness:
	bash scripts/check_external_gate_readiness.sh

test:
	bash scripts/run_tests.sh

laptop-smoke:
	STUDY_PROFILE=laptop_tiny LOB_FORGE_MAX_PROCESS_MEMORY_GB=6 bash scripts/run_60day_expected_edge_study.sh

cloud-package:
	bash scripts/package_cloud_handoff.sh

modal-study:
	bash scripts/run_modal_expected_edge.sh

kelly-candidate:
	bash scripts/run_kelly_candidate_search.sh
