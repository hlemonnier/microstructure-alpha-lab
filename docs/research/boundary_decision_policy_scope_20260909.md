# Probability reproduction and method-specific decision priors

The completed proper-score experiment reproduces all six original neural state dictionaries and every original probability, label, decision clock and training prior exactly. Under the shared primary `forecast_3600` rule, decision priors and all classification metrics also reproduce exactly. The twelve asset/date/representation checks are recorded in the [proper-score evidence](boundary_proper_score_evidence_20260908.json).

The secondary label `registered` preserves each method's registered rule; it does not mean one universal prior across all historical sources. The original observation and combined neural controls inherited

    decision_prior = (historical_training_prior + previous_noon_label_prior) / 2.

The new proper-score cases use

    decision_prior = historical_training_prior.

Both formulas are verified against the actual saved arrays and the previous-noon validation labels in all twelve comparisons. The legacy formula originates in `scripts/run_boundary_regime_screen.py`, and is inherited by the observation-context and combined-feature neural sources. The new formula is explicit in the completed proper-score runner. Both use past information.

Because a balanced decision is `argmax_c probability[c] / decision_prior[c]`, identical probabilities can produce different secondary decisions when their denominators differ. Natural log loss and ordinary probability-argmax accuracy remain unchanged. This is not a parameter or probability-reproduction failure.

The initial postprocessing helper incorrectly required all metrics to match under both tags. Its failed assertion and original helper are preserved. The corrected check requires exact probabilities, exact primary decisions and metrics, and exact verification of each distinct secondary prior formula. No model, forecast array, decision array, score or frozen market protocol was changed.

Within the proper-score family, every loss and its fresh log-loss control uses the same decision rule, so those comparisons isolate the loss. Cross-family comparisons use the common primary forecast-prior procedure. Secondary method-specific comparisons must be described as comparisons of complete procedures, not effects caused solely by a learner or loss.

The prospective twenty-date audit has the same distinction: its new observation anchors use the historical training prior for secondary `registered` decisions, while all retained controls keep their original saved rules. Its primary forecast-prior procedure and required original parameter/probability reproduction are unchanged. The prepared attention family also shares one rule across its new branches. This clarification changes no candidate, cohort, model budget, selection rule or success gate.
