# Quote-renewal synthetic prototype

The reflected timestamp-block renewal prototype passed 44 synthetic E2E steps. The parent independently verified all 190 artifact hashes and the final module/runner hashes; scoped Ruff also passed. This verifies the implemented model and causal plumbing on constructed inputs. There is no market-trained model or predictive improvement result.

The [design](boundary_quote_renewal_design_20261011.md) was committed before implementation. A final pre-implementation preparation receipt at `artifacts/quote-renewal-e2e-20261010T233927157597Z/evidence.json` preserves 56 independently checked files and has SHA256 `9c24160c210aa8854dee9c0194d6e8fceeef0326547bf4e21896e379c32c050b`. All preparation drafts and four complete E2E attempts remain preserved locally. No unit tests were written.

Checks cover publisher-time first/last ordering, the exact 100ms/5100ms stopping clocks, strict predecision features, age-conditioned component probabilities, censor terms, monotone regularized EM objectives, joint duration/price/state marks, arbitrary query price fractions, exact target ties, same-block entry/exit, reflected versus unreflected mass, sequential versus cached reflection, positive support, save/reload and query partition/order. Thirteen expected rejection receipts cover malformed inputs and bounded numerical/resource failures. An analytic homogeneous kernel checks the simulation at a predeclared wiring tolerance; it is not a market convergence certificate.

The final synthetic fit took 0.251 seconds, with 36.2 MB peak process RSS; its 6,144-path cache took 0.074 seconds. Maximum query probability shifts were approximately 0.04628 across seeds and 0.04520 after doubling the 256-path budget. That budget is insufficient evidence of numerical stability. [Training-only feasibility](boundary_quote_renewal_feasibility_design_20261011.md) must freeze sources, windows, seeds, path budgets, tolerances and supervised resource limits before market use. The CLI currently explicitly accepts synthetic manifests only.

Reproduce from the repository root:

```sh
.venv/bin/python scripts/run_quote_renewal_e2e.py
```

The command creates a fresh timestamped artifact directory with commands, outputs, failures, fixtures and a hash inventory. [Final evidence](boundary_quote_renewal_synthetic_evidence_20261011.json) refers to `artifacts/quote-renewal-e2e-20261010T234923912741Z/`; the full generated artifacts remain local. Its SHA256 is `76fd68cf3dbb7fb4104613e434f65b3e20c8673b7c34a13096a346a054124c22`.

Frozen verified source hashes:

- `src/lob_forge/boundary_quote_renewal.py`: `9e18062ea8be79b7aa15ef664786ef753869a45b42edecaa20b9e08ae696ad7e`.
- `scripts/run_quote_renewal_e2e.py`: `dfdb523727f35666fc0bd996b1cadd7e0f7e10722c90b09454703d4f5f70223d`.

The reflected positive-price boundary is a modeling assumption, not a claim about exchange behavior near zero. The observed state is coarse and the conditional law stationary; the two-component mixture resets at each new block. A [persistent latent-state extension](boundary_latent_renewal_research_20261011.md) remains conceptual and has unresolved likelihood and observation-support requirements.
