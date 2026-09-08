# Delayed supervised adaptation: completed development result

No adaptive procedure beats the strongest existing balanced-accuracy control. Full-network updates and replay improve some probability log losses, but the gain sought in predictive accuracy remains unconfirmed.

The complete study contains thirty trajectories on June 3, 7 and 11: eighteen full-network adaptations, six bias-only adaptations and six frozen reproductions. Every adaptive trajectory makes 180 updates, giving 4,320 optimizer steps. All three dates finish before the 1,236 fixed panels are scored. The 42,420 decisions are the same repeatedly exposed development observations.

| Procedure, with the fixed one-hour forecast-prior policy | Balanced accuracy | Natural log loss |
| --- | ---: | ---: |
| Existing strongest `deep500_hgb_old_neural` | 57.3074% | 0.761756 |
| Observations bias adaptation + `deep500_hgb` | 57.2945% | 0.760295 |
| Combined full-network replay + matched HGB | 57.2721% | 0.758226 |
| Combined frozen neural + matched HGB | 57.0984% | 0.760819 |
| Combined faster full-network replay + matched HGB | 56.5905% | 0.762464 |

The best full-network blend gains 0.1737 percentage points over its own frozen combined-feature blend, with positive asset averages but one slightly negative date. It remains 0.0353 points below the strongest control. The best adaptive source is the bias-only blend, itself 0.0130 points below that control. Faster replay adaptation degrades accuracy. Some natural probability log losses improve under the tested settings; this is not an overall accuracy discovery.

At each minute, the model uses only the preceding 30-minute window whose actual target quote has arrived, plus a separate 100ms observation allowance. The new parameters become available ten seconds later. The previous state supplies all earlier forecasts. Historical replay uses 6,144 unique balanced asset/class examples from the original past training cohort. The largest observed update takes 0.110829 seconds, within the fixed allowance; this is update timing, not end-to-end trading latency certification.

The original normalizer and training priors stay fixed. For network logits `z`, inference returns probabilities proportional to `exp(z) * original_prior`. Recent balanced cross entropy therefore uses logits `z + log(original_prior) - log(recent_prior)` and weights `N / (6 * asset_class_count)`. Replay uses unweighted cross entropy on original logits because its bank is already balanced. Exact full-cohort gradient checks verify the accumulation and weighting.

The initial attempt completed one frozen reproduction and then stopped at the first adaptive release-clock check, before any optimizer update or new score. Canonical parquet release times were binary64 values. The [adapter amendment](boundary_prequential_release_clock_evidence_20260908.json) accepts only finite, exactly integral values within the consecutive-integer range and never rounds. All 2,418,881 resolved release values in 28 source sessions round-trip exactly. The first attempt remains preserved; the complete restart gives 32 total trajectory attempts, including that initial failure and reproduction.

The [frozen restart protocol](boundary_prequential_screen_release_clock_20260908.json) and [result evidence](boundary_prequential_evidence_20260908.json) bind the study to summary SHA256 `bb2746b56d6d62b0560788dc33a3eb459a734505b248a0f537e1d6520828e69b`. Six frozen forecast paths reproduce their existing checkpoints exactly. The full dependency-complete suite passes 633 direct tests. Neither test success nor reuse of these dates supplies independent accuracy confirmation. The separately registered continuous-memory experiment follows this completed study.
