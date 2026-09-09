# Alternative quote currencies reveal omitted BTC activity

In the original first-fold training window, May 29–June 1, 2023, BTC/TUSD traded **1.855569 times the BTC quantity** and reported **3.819909 times the trade count** of BTC/USDT spot. BTC/TUSD exceeds BTC/USDT base quantity on each of the four dates. The existing spot feature source uses BTC/USDT. This is a concrete omitted-information hypothesis, not an accuracy result.

Binance announced that BTC/TUSD would retain zero maker and taker fees from March 22, 2023 UTC, while BTC/USDT and BTC/BUSD returned to standard fees. ETH/TUSD retained zero maker fees. The announcement motivates checking the period-specific source mix; it does not prove that fee-free volume contains more predictive information. [Official announcement](https://www.binance.com/pt-BR/support/announcement/detail/be13a645cca643d28eab5b9b34f2dc36).

| Pair | Four-day base quantity | Base unit | Reported trades | Active minutes / 5760 |
| --- | ---: | --- | ---: | ---: |
| BTCBUSD | 19934.27815000 | BTC | 566,975 | 5,760 |
| BTCTUSD | 298521.62573000 | BTC | 12,506,908 | 5,760 |
| BTCUSDT | 160878.78769000 | BTC | 3,274,138 | 5,760 |
| ETHBTC | 115047.51850000 | ETH | 196,875 | 5,744 |
| ETHBUSD | 196240.09370000 | ETH | 562,305 | 5,760 |
| ETHTUSD | 17926.03200000 | ETH | 95,803 | 4,762 |
| ETHUSDT | 1176011.51220000 | ETH | 1,742,176 | 5,760 |
| TUSDUSDT | 396055268.00000000 | TUSD | 239,683 | 5,740 |

The ETH result is different: ETH/TUSD has only 1.5243% of ETH/USDT base quantity. ETH/BUSD and ETH/BTC have 16.6869% and 9.7829% respectively. BTC/TUSD activity is the strongest source-coverage lead from this inventory; the ETH/BTC market is a possible distinct relative-price channel, not a volume leader.

TUSD/USDT minute closes range from 0.9992 to 0.9998 in this window. Treating TUSD as exactly one USDT would introduce a material price-basis error relative to the five-second forecast target. Future features must either retain within-pair returns and signed quantities, or use explicitly observed, past-only conversion prices with age and missingness indicators. Reported zero-fee trade counts may include activity with little additional information; only matched predictive tests can resolve that.

All 32 registered archives were available and matched their official SHA256 checksums. Their 46,080 minute rows passed complete-clock, candle-range, quantity-consistency and ZIP-integrity checks. The total compressed archive size was 1,840,758 bytes. These candles are used only for this source inventory; a candle containing time after a forecast decision cannot become a predictor. [Official data schema](https://github.com/binance/binance-public-data).

The counted-depth experiment remains unchanged and its new assessment metrics remain sealed. The next source feasibility step is a bounded inventory of BTC/TUSD, TUSD/USDT and ETH/BTC aggregate-trade archives on the existing exposed source dates. Any predictive study must be frozen separately, preserve the original target and rows, and follow the current study and its broader-evaluation priority. No independent confirmation data were opened and no additional model was fitted.

Full inventory SHA256: `7b41cc00a3369cfec3a850e6c84db4100bc20966d83f89eec5419bdfa3f3d862`. Tracked evidence SHA256: `99c7c018a33eccea3132fb793b326af9cc8dcf99aac9e8ce958c34f64907c139`. Protocol SHA256: `8af0fbdd65e819f86052f6d4918f449090e6a3efd81f863b31c6b76c0dd0e2c6`.
