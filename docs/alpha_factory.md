# Alpha Factory Operating Rules

This project is now organized around falsifiable expected-edge hypotheses, not model demos.

## Hypothesis Format

Every candidate should be stated as:

```text
H_i = (U, D, F, y, h, L, M, pi, C, R, A)
```

where:

- `U`: universe, venue, contracts,
- `D`: data source and date range,
- `F`: feature set available before decision time,
- `y`: prediction or economic target,
- `h`: horizon,
- `L`: latency model,
- `M`: model class,
- `pi`: decision rule,
- `C`: cost model,
- `R`: risk/position-sizing rule,
- `A`: acceptance criterion.

Example:

```text
On Binance USD-M BTCUSDT/ETHUSDT, using quote/trade/depth-band features available before t,
a model can predict 5-second executable taker edge after 1-second latency such that OOS net
mean return per trade is positive after costs, across at least 70% of test folds, with no
single fold contributing more than 40% of total PnL.
```

## Primary Target

Classification labels remain useful diagnostics, but trading selection should use executable economics.

Long taker gross return:

```text
r_long = (future_bid - entry_ask) / entry_ask
```

Short taker gross return:

```text
r_short = (entry_bid - future_ask) / entry_bid
```

The decision rule should act on predicted net edge after costs:

```text
side = +1 if mu_long > mu_short and mu_long > tau
side = -1 if mu_short > mu_long and mu_short > tau
side = 0 otherwise
```

where `tau` is selected only on validation.

## Rejection Gates

A candidate is rejected if any of these fail:

- OOS net PnL is positive after realistic costs.
- At least 20 OOS test folds are available for a serious claim.
- Median fold net PnL is positive.
- Positive fold rate is at least 70%.
- No single fold contributes more than 40% of total OOS net PnL.
- Break-even fee clears assumed all-in costs with a safety multiple.
- The candidate beats `always_flat` on the validation-selected economic objective.
- The candidate does not rely on one day, one symbol, one regime, or one latency assumption.
- The complex model beats the simple threshold baseline.

## Final Holdout Candidate Freeze

Before evaluating any fixed threshold rule on the immutable holdout, freeze the validation-selected rule into candidate JSON and pre-register that exact candidate hash in the holdout manifest:

```bash
PYTHONPATH=src python3 -m lob_forge.cli freeze-threshold-candidate results/current/<threshold_walk_forward_result.csv> \
  --output results/final_holdout/<candidate>.json \
  --sort-by validation_net_pnl \
  --taker-fee-bps 0

PYTHONPATH=src python3 -m lob_forge.cli create-holdout-manifest data/processed/<feature-source>.csv \
  --output results/holdout_manifests/<candidate>.json \
  --split-column source_date \
  --holdout-values <final_holdout_dates> \
  --candidate-json results/final_holdout/<candidate>.json
```

For the expected-edge ridge model, freeze the model weights from manifest-filtered development rows and use an edge threshold selected from the walk-forward artifact:

```bash
PYTHONPATH=src python3 -m lob_forge.cli freeze-edge-candidate data/processed/<feature-source>.csv \
  --holdout-manifest results/holdout_manifests/<development>.json \
  --output results/final_holdout/<edge_candidate>.json \
  --walk-forward-artifact results/current/<edge_walk_forward_result.csv> \
  --sort-by validation_net_pnl \
  --taker-fee-bps 0

PYTHONPATH=src python3 -m lob_forge.cli create-holdout-manifest data/processed/<feature-source>.csv \
  --output results/holdout_manifests/<edge_candidate>.json \
  --split-column source_date \
  --holdout-values <final_holdout_dates> \
  --candidate-json results/final_holdout/<edge_candidate>.json

PYTHONPATH=src python3 -m lob_forge.cli final-holdout-edge data/processed/<feature-source>.csv \
  --holdout-manifest results/holdout_manifests/<edge_candidate>.json \
  --candidate-json results/final_holdout/<edge_candidate>.json \
  --output results/final_holdout/<edge_result>.json \
  --explicit-final-evaluation
```

The frozen edge candidate stores the selected threshold, feature list, standardizer, ridge weights, cost assumptions, source data path, and development manifest hash. The final command refuses to run unless the holdout manifest pre-registers the candidate hash.

## Multiple Testing

Every tried candidate counts:

```text
features x thresholds x horizons x latency values x regimes x fee assumptions x models
```

Use:

```bash
PYTHONPATH=src python3 -m lob_forge.cli pvalue-correction results/current/pvalues.csv
```

The CSV must contain:

```text
hypothesis_id,metric,p_value
```

The output includes Bonferroni and Benjamini-Hochberg adjusted p-values.

For expected-edge study runs, do not build this file from audit filenames by hand. Refresh the candidate registry after all result/audit artifacts are present and ask the registry writer to emit candidate-linked p-values:

```bash
PYTHONPATH=src python3 -m lob_forge.study_registry \
  --plan results/current/run_plan.json \
  --result-dir results/current \
  --output results/current/candidate_registry.jsonl \
  --pvalues-output results/current/pvalues.csv
PYTHONPATH=src python3 -m lob_forge.cli pvalue-correction \
  results/current/pvalues.csv > results/current/pvalue_corrections.csv
```

The expected-edge verifier requires each completed candidate/config in `candidate_registry.jsonl` to have a matching `config_sha256` row in both `pvalues.csv` and `pvalue_corrections.csv`; this makes the multiple-testing correction family explicit.

## Capacity Screen

The current archive can support only a conservative taker-capacity screen:

```text
capacity_per_signal <= min(
  participation_rate * trade_notional,
  top_book_fraction * displayed_top_book_notional,
  optional max_notional
)
```

Run:

```bash
PYTHONPATH=src python3 -m lob_forge.cli capacity data/processed/<feature-source>.csv \
  --feature microprice_deviation \
  --threshold 0.1 \
  --participation-rate 0.01 \
  --top-book-fraction 0.05
```

This is not a production capacity model. Passive capacity still requires queue position, cancellations, partial fills, and live/paper fill calibration.

## Kelly Sizing Gate

Fractional Kelly is disabled unless OOS fold variance is stable enough to pass a separate gate. Run:

```bash
PYTHONPATH=src python3 -m lob_forge.cli kelly-variance-gate results/current/<artifact>.csv \
  --column validation_net_pnl \
  --min-observations 20 \
  --window-size 5 \
  --max-variance-cv 0.5
```

The command estimates variance across non-overlapping fold windows, reports the coefficient of variation of those window variances, and exits non-zero when the evidence is too thin or unstable. The gated Kelly helper returns zero notional unless this report passes.

The aggregate evidence gate is stricter than variance alone. It scans promoted local16 strategy artifacts plus reproducible Kelly candidate-search artifacts and requires the same nonzero-cost candidate to pass audit acceptance and variance stability. The default cost floor is `0.05` bps, so zero-fee artifacts cannot enable Kelly.

Current local evidence has one narrow Kelly-eligible candidate: `results/kelly_candidate_search/BTCUSDT_5000ms_fee_0p05_balanced_edge.csv`. It uses a predeclared balanced threshold grid from `scripts/run_kelly_candidate_search.sh`, passes the audit at 20 folds, has weighted break-even fee `0.100291` bps against the `0.100000` bps safety requirement, and passes variance stability with `variance_cv=0.487520055465` against the `0.5` limit. This is still a local16 capped research artifact, not permission to size real capital before the full cloud study and live fill validation finish.

## Sequence Model Readiness Gate

Do not run Transformer/TCN/LOB-CNN experiments until both the economic baseline and the true-L2 tensor path are ready:

```bash
PYTHONPATH=src python3 -m lob_forge.cli model-readiness-gate \
  --model sequence_transformer \
  --baseline-audit results/current/btc_full_day_edge_zero_fee_audit.csv \
  --l2 data/normalized_l2/okx/BTC-USDT-SWAP/2023-05-16.csv \
  --min-fold-count 20 \
  --min-l2-rows 1000
```

Use `data/normalized_l2/bybit/BTCUSDT/2023-05-16.csv` when the Bybit history-data resolver/importer is the first true-L2 path available. The aggregate `evidence-gates` command checks both OKX and Bybit default BTC smoke paths unless explicit repeatable `--l2` arguments are supplied.

The gate checks:

- optional model dependency availability,
- accepted baseline audit with enough folds,
- normalized L2 rows with snapshot and delta events,
- sequence or update IDs,
- bid and ask sides,
- no detected sequence gaps or crossed-book states,
- true crypto L2 rather than FI-2010 unless `--allow-fi2010` is explicitly set for a sanity benchmark.

For serious L2 sequence experiments, create a holdout manifest against the source L2 CSV first and pass it to the runner. The command verifies the manifest, writes a filtered development L2 CSV, and trains only on the development rows:

```bash
PYTHONPATH=src python3 -m lob_forge.cli l2-sequence-experiment \
  --model sequence_transformer \
  --baseline-audit results/current/btc_full_day_edge_zero_fee_audit.csv \
  --l2 data/normalized_l2/bybit/BTCUSDT/2023-05-16.csv \
  --output results/model_experiments/sequence_transformer_results.csv \
  --holdout-manifest results/holdout_manifests/bybit_l2_sequence_holdout.json \
  --development-l2-output results/model_experiments/development_l2/sequence_transformer_development_l2.csv \
  --checkpoint-path results/model_experiments/checkpoints/sequence_transformer.pt \
  --predictions-output results/model_experiments/predictions/sequence_transformer_predictions.csv \
  --device auto \
  --class-weighting balanced \
  --min-fold-count 20 \
  --min-l2-rows 1000

PYTHONPATH=src python3 -m lob_forge.cli l2-sequence-experiment \
  --model sequence_tcn \
  --baseline-audit results/current/btc_full_day_edge_zero_fee_audit.csv \
  --l2 data/normalized_l2/bybit/BTCUSDT/2023-05-16.csv \
  --output results/model_experiments/sequence_tcn_results.csv \
  --holdout-manifest results/holdout_manifests/bybit_l2_sequence_holdout.json \
  --development-l2-output results/model_experiments/development_l2/sequence_tcn_development_l2.csv \
  --checkpoint-path results/model_experiments/checkpoints/sequence_tcn.pt \
  --predictions-output results/model_experiments/predictions/sequence_tcn_predictions.csv \
  --device auto \
  --class-weighting balanced \
  --min-fold-count 20 \
  --min-l2-rows 1000
```

The aggregate evidence gate now inspects these artifact rows; they must report the expected `model_name`, selected `l2_path`, `readiness_passed=1`, `dependency_available=1`, and `pipeline_completed=1`. Sequence artifacts also record holdout manifest path/hash, the filtered development L2 path, holdout row counts, minibatch size, early-stopping patience, best epoch, selected device, class weighting, scheduler gamma, checkpoint path, prediction-export path, Brier/ECE calibration metrics, confusion matrices, and stateful test-set economic smoke fields. `acceptance_passed` remains separate and should only become true when an explicit predictive/economic threshold is defined and met.

Before treating a neural sequence run as a final candidate, freeze the manifest-filtered artifact and checkpoint. The freeze step records the checkpoint hash, development L2 hash, development-only standardizer values, and stateful economic settings; the final command then loads the frozen checkpoint and evaluates only manifest-selected holdout L2 rows without refitting preprocessing. Any final-holdout economic CLI values must match the frozen candidate, so they act as assertions rather than post-freeze tuning inputs:

```bash
PYTHONPATH=src python3 -m lob_forge.cli freeze-sequence-candidate \
  results/model_experiments/sequence_transformer_results.csv \
  --output results/final_holdout/sequence_transformer_candidate.json

PYTHONPATH=src python3 -m lob_forge.cli create-holdout-manifest \
  data/normalized_l2/bybit/BTCUSDT/2023-05-16.csv \
  --output results/holdout_manifests/sequence_transformer_final_holdout.json \
  --split-column exchange_timestamp \
  --holdout-values <final_holdout_exchange_timestamps_or_session_ids> \
  --candidate-json results/final_holdout/sequence_transformer_candidate.json

PYTHONPATH=src python3 -m lob_forge.cli final-holdout-sequence \
  data/normalized_l2/bybit/BTCUSDT/2023-05-16.csv \
  --holdout-manifest results/holdout_manifests/sequence_transformer_final_holdout.json \
  --candidate-json results/final_holdout/sequence_transformer_candidate.json \
  --output results/final_holdout/sequence_transformer_final_holdout_result.json \
  --predictions-output results/final_holdout/sequence_transformer_final_predictions.csv \
  --explicit-final-evaluation
```

The dependency-free self-supervised smoke artifact is generated with:

```bash
PYTHONPATH=src python3 -m lob_forge.cli l2-pretraining-smoke \
  --l2 data/normalized_l2/bybit/BTCUSDT/2023-05-16.csv \
  --output results/model_experiments/self_supervised_pretraining.csv \
  --depth 5 \
  --window 4
```

Current local state: the Bybit true-L2 smoke path is ready, the masked reconstruction pretraining artifact exists, and local Torch smoke artifacts exist for `sequence_transformer` and `sequence_tcn` under `results/model_experiments/`. These prove the gated training/evidence path and artifact contract, not production model edge. Current neural outputs are tiny-sample smoke artifacts and should not be promoted as alpha.

The sequence runner also has a deterministic ablation axis. Start with a dry run so the full matrix is visible before any compute is spent:

```bash
ABLATIONS=baseline,no_class_weighting,no_lr_scheduler,short_window,shallow_depth \
SEEDS=7,11,13 \
DRY_RUN=1 bash scripts/run_l2_sequence_experiments.sh
```

Each ablation keeps the same baseline/L2 readiness gates, optional holdout-manifest filtering, checkpoint, prediction-export, calibration, and stateful-economic fields as the baseline artifacts.

## Result Auditing

Saved walk-forward CSVs can be audited with:

```bash
PYTHONPATH=src python3 -m lob_forge.cli audit-results results/current/<artifact>.csv
```

The audit reports:

- fold count,
- total trades,
- inference grain,
- total net PnL,
- positive fold rate,
- median fold net PnL,
- largest positive fold share,
- break-even fee,
- median fold mean/median net bps,
- fold profit factor,
- max fold drawdown in raw PnL units,
- mean fold Sharpe-per-trade summary,
- fold-bootstrap mean bounds,
- one-sided HAC/Newey-West mean-positive p-value proxy at the declared inference grain,
- acceptance/rejection reasons.

## Calendar Splits

Row-count walk-forward remains useful for quick sanity checks, but serious runs should group by market dates:

```bash
PYTHONPATH=src python3 -m lob_forge.cli create-holdout-manifest data/processed/<feature-source>.csv \
  --output artifacts/holdout_manifests/<feature-source>.json \
  --split-column source_date \
  --holdout-values <final-date> \
  --source-root "$PWD"
PYTHONPATH=src python3 -m lob_forge.cli calendar-walk-forward data/processed/<feature-source>.csv \
  --holdout-manifest artifacts/holdout_manifests/<feature-source>.json \
  --train-days 20 \
  --validation-days 5 \
  --test-days 5 \
  --taker-fee-bps <fee>
```

The output includes the exact train, validation, and test date groups for each fold. Current four-day artifacts use a small 2/1/1-day split only to verify the machinery; they are not alpha evidence.

## 60-Day Expected-Edge Study

The review's highest-value next experiment is scripted as:

```bash
bash scripts/run_60day_expected_edge_study.sh
```

Laptop defaults:

- profile: `laptop_tiny`,
- symbol: `BTCUSDT`,
- dates: `2023-05-16` to `2023-05-17`,
- horizon: `5000` ms,
- latency: `1000` ms,
- fees: `0 0.1 0.5` bps,
- depth disabled and quote rows capped.

This default is only a laptop proof-of-pipeline. The full review experiment remains `STUDY_PROFILE=cloud_full` and is intentionally a large data job for a 64-128 GB cloud machine. The capped `local16_60day` profile is blocked behind `CONFIRM_LOCAL16=1` and should be planned first with `PLAN_ONLY=1`.

## Hard Limits Of Current Data

The current Binance Vision archive supports quote/trade/depth-band microstructure research. It does not support:

- true historical multi-level LOB tensors,
- queue-position reconstruction,
- production maker fill simulation,
- strong passive capacity claims.

Those require live diff-depth capture with sequence validation, snapshots, deterministic replay, and paper/live fill calibration.
