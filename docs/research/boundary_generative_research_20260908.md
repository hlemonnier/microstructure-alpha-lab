# Generative mechanisms for the accuracy frontier

This note records primary-source research while the native-flow experiment is sealed. It creates no additional model fit, parameter sweep or promotion. The objective remains the existing exact five-second, three-class target; all May 24–June 12 dates remain exposed development data.

## What the papers actually establish

| Approach | Evidence inspected | Implication for this repository |
| --- | --- | --- |
| Queue-reactive and Hawkes models | [Huang, Lehalle and Rosenbaum](https://arxiv.org/abs/1312.0563) model state-dependent order arrivals. [Wu et al.](https://arxiv.org/abs/1901.08938) combine book state and past excitation on Eurex data. | A plausible mechanism for conditioning future activity on liquidity and recent flow. Realistic event timing alone does not establish better five-second class prediction. |
| Multidimensional deep queue-reactive model | [Bodor and Carlier, section 4.6](https://arxiv.org/html/2501.08822v1) report 0.63 balanced accuracy versus 0.54 for their DeepLOB comparator on Bund futures. Their target uses smoothed prices across 500 events, and their data distinguishes provision, cancellation and market activity. | The reported nine-point difference is motivation, not a transferable gain estimate. Our target, venue, observation process and training protocol differ. Their market-event labels are richer than aggregated absolute-quantity updates. |
| Hawkes timing plus continuous-time price dynamics | [Raffaelli et al., sections 3–4](https://link.springer.com/article/10.1007/s10203-026-00570-z) evaluate Bitfinex BTC/USD and ETH/USD. Their task predicts the next mid-price-change direction; experiments use short testing windows after recent training. Trading simulations omit costs. | Potentially useful for conditional movement intensity. Their binary event target cannot be compared directly with a fixed five-second three-class score. The stated timing statistic is predicted interval divided by realized interval; its ideal value is one, so simply minimizing it is not a valid error objective. |
| Diffusion trajectories | [Backhouse et al.](https://arxiv.org/abs/2509.05107) use structured LOB representations and diffusion inpainting, evaluating generative realism on LOB-Bench. [DiffLOB](https://arxiv.org/abs/2602.03776) explicitly conditions counterfactual trajectories on future regimes. | A past-conditioned trajectory generator is a legitimate candidate. Realized future trend, volatility or imbalance must never be supplied at inference. Counterfactual conditioning and realistic simulated marginals do not themselves establish forecasting accuracy. |
| Microprice | [Stoikov](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694) defines a limit of conditional expected future mid-prices using book state. | A state-dependent fair-price correction could provide a compact auxiliary forecast. Expected price alone is insufficient to recover the probabilities of our three thresholded outcomes. |

## Observation model before architecture

The native Bybit and Binance depth messages contain batches of absolute quantity revisions. A decrease cannot be uniquely separated into cancellations and executions. Several underlying order events may occur inside one published update, including replenishment that offsets removal. Capture timestamps additionally include the recording path. Treating each published price revision as a single economic point-process event would impose an unsupported observation model.

For a genuine point process, the log likelihood includes both event log intensities and the integrated intensity over elapsed exposure. Omitting the exposure term rewards arbitrarily high intensities. For batched data, a model of observed interval counts or net/gross revision masses is more defensible than inventing individual order timestamps and types. These are modeling judgments for this dataset, not claims that the cited methods fail on their own data.

Finite REST snapshots impose another information boundary. All quantities inside the initial visible range are known, including absent price levels. Beyond that range, an absolute update reveals its own price's quantity, but does not establish the absence of unreported orders at neighboring prices. A reconstructed sparse dictionary must not silently become a certified full depth curve after the market leaves the snapshot range.

## Execution order

1. Finish the frozen native-flow screen with the matched BBO and deeper-flow controls. Reveal all three dates together.
2. Measure target-venue depth coverage using recorded capture times and explicit snapshot frontiers. Preserve all original decision rows.
3. If coverage supports a useful test, compare the same predictive learners with target-venue depth on identical past-only training and assessment partitions. If full depth is not certifiable, define explicitly partial observed-liquidity features with knowledge masks before testing them.
4. A subsequent generative candidate must output a distribution over the same five-second terminal change, integrate over unavailable future states, and beat the strongest direct supervised comparator on the same rows. Freeze that experiment separately; this note is not a model protocol.

RL becomes relevant when the objective includes actions, inventory and counterfactual execution outcomes. Under the user's predictive-accuracy priority, its simulator assumptions would add a separate problem before a forecast gain is established.
