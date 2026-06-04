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
