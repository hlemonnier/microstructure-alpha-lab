# Conversion-price microstructure: a concrete representation lead

On the original May 29-June 1 training dates, TUSD/USDT trades occupy an observed price grid with minimum spacing 0.0001 USDT/TUSD, approximately one basis point. That step is roughly 27 times the median current-quote analogue of the BTC classification barrier. The actual target uses a future entry quote, which this diagnostic did not read. Observed spacing is not an exchange tick-rule certification or an estimate of the true FX-price error.

The fixed transform `adjusted_rate = traded_rate - 0.00005 * taker_sign` uses +1 for an aggressive buy and -1 for an aggressive sell. The half-spacing is fixed from the completed training-only resolution audit. Adjacent opposite-side trades within one second are separated by one observed spacing in at least 99.9% of cases on each date. This strongly supports a bid-ask-bounce interpretation for these training observations. It does not establish the actual executable quote or the latent efficient price.

| Training date | Median spacing / known BTC barrier | Opposite-side one-spacing fraction | Adjusted / raw sampled increment variance |
| --- | ---: | ---: | ---: |
| 2023-05-29 | 27.892207 | 99.976836% | 0.000574803 |
| 2023-05-30 | 27.776638 | 100.000000% | 0.000364232 |
| 2023-05-31 | 27.120528 | 99.914621% | 0.001411320 |
| 2023-06-01 | 26.882903 | 99.904626% | 0.001058948 |

The adjustment reduces observed one-second conversion increment variance by 99.86-99.96% on these four training dates. Lower variation alone does not prove a better forecast: oversmoothing could erase real movements. The concrete next test is to compare the unadjusted source plus its observed trade side with the adjusted source plus the same side, keeping both conversion-only and native-market controls. This separates new side information from the value of the explicit midpoint proxy. The current 36-model screen remains unchanged.

Every diagnostic input was verified before and after execution. Both audits read source trades and known quote columns only, fitted no predictive model and read no target labels, future entry quotes or independent confirmation data.

The broader literature supports examining measurement and timing carefully, but does not supply a transferable accuracy claim. [Hasbrouck, Price Discovery in High Resolution](https://people.stern.nyu.edu/jhasbrou/Research/HRVAR/HRVAR08.pdf) finds that coarse sampling can conceal information relationships visible at finer resolution in two US equities. Its information-share results do not measure our five-second three-class accuracy. [Buccheri, Bormetti, Corsi and Lillo](https://arpi.unipi.it/bitstream/11568/998290/7/Comments%20on%20Price%20Discovery%202019.pdf) distinguish microstructure noise from delayed price adjustment and treat sparse observations as missing measurements in a state-space model. A future latent-price filter would need causal filtering, stable transition dynamics and training-only parameter estimation; their simulation results do not establish an improvement here.

A recent [fee-structure study](https://link.springer.com/chapter/10.1007/978-3-032-18109-1_13) is particularly easy to overinterpret. Its main regression explains contemporaneous price changes, and the authors explicitly report negligible one-step-ahead predictive power. Its abstract mixes a roughly 5% starting R-squared with a 21.55% endpoint, while the detailed results assign the hourly comparison to 5.03%-11.15% and the ten-minute comparison to 4.04%-21.55%. Its methods also alternate between the March 15 announcement and March 22 implementation. These results motivate controls; they cannot justify a predicted accuracy gain or replace the official fee-change date.

The independent twenty-date confirmation reservation remains unopened for market analysis. No substantial predictive gain has been confirmed.

Evidence SHA256: `58063c29879c0646a8e2993963118305923934b8fc3019a6b1ad1e895e95f5f6`. Resolution result SHA256: `9d204c6aee1ac9ab91d338e5cc0a7a260a194ac73414776e312251d05faeb747`. Bounce result SHA256: `1f6965178e641b5e7954a5c4a1fb88e2e533ef62bbd7705920709499b0fb7734`.
