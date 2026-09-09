# Alternative quote-currency feature preparation

All 14 exposed source dates (May 29 through June 11, 2023) passed the fixed source and conversion checks. The preparation retained 28 target sessions, 56 feature groups at 100/500 ms delays, and 2,419,034 original decision rows. No model was fitted and no predictive score was computed.

BTC/TUSD is converted using observed TUSD/USDT trades; ETH/BTC uses observed BTC/USDT trades. Rates must be strictly available before the decision after the stated delay and no more than 60 seconds old. Invalid converted basis observations cannot update the basis smoothing state. Native trade flows and returns retain their native-market meaning.

All 348 pinned preparation inputs and each daily artifact inventory were verified. Each source had contiguous aggregate IDs and its first/last trade within five minutes of the day boundaries. Every cache passed exact native-field replay, identity-currency conversion, truncated-source prefix and Parquet reload checks. The original June 1 training preflight was reused exactly; thirteen additional daily workers completed within their fixed resource budgets.

Preparation elapsed time: 157.381651 seconds. Maximum sampled worker RSS: 1,405,272,064 bytes. Lowest conversion availability among the 56 groups: 98.847142%.

| Native or conversion source | Aggregate-trade records |
| --- | ---: |
| BTCTUSD | 31,799,309 |
| TUSDUSDT | 1,016,170 |
| ETHBTC | 463,478 |
| BTCUSDT | 10,224,573 |

These are source-integrity results, not predictive gains. Zero-fee trading volume may include uninformative activity. The next predictive study must compare conversion-only controls with BTC/TUSD and ETH/BTC additions on the same rows. Historical publisher timestamps do not certify live arrival latency, and last-trade availability does not certify a fresh executable quote.

The twenty reserved independent confirmation dates remain unopened for market analysis. This preparation does not change the substantial-gain requirement or consume a confirmation round.

Feature manifest SHA256: `611206220647be75b53d8af6b204285b1c9f9c23f6e373f2fd3a41c9e737cd4f`. Evidence SHA256: `f82228c20382176b1ecf7be9344233f896d0555069f0deb041c06532cc08287a`.
