# Stock Market Prediction - RNN

## Data Gathering

* Daily historical stock data were collected for SBI (SBIN), LIC (LICI), HAL, NTPC, and ONGC through Angel One SmartAPI, covering January 2025 to January 2026.  
* The dataset contains opening, highest, lowest, and closing prices, along with trading volume.  
* Monthly CSV files were organised by company and year, with the collection logs reporting 269 daily records per company.

## EDA

#### Daily Closing Prices

![](/images/closing-prices.png)

* SBI exhibited a sustained upward trend after March 2025, rising from approximately ₹700 to above ₹1,050 by January 2026.
* LIC and HAL recovered from early declines and reached peaks around June–July 2025, followed by lower prices in the subsequent months.
* NTPC and ONGC fluctuated within comparatively bounded price ranges for much of the period, with both showing an upward movement in January 2026.

#### Closing Prices Indexed

![](/images/closing-prices-indexed.png)

* Indexing initial closing prices to 100 enables comparison of relative price changes across the five companies.
* SBI recorded the largest overall increase, ending near 135—approximately 35% above its initial price—while LIC ended near 92, indicating an approximately 8% decline.
* HAL fell to approximately 74 before recovering above 120 mid-period; ONGC and NTPC ended above their initial levels, with gains of roughly 14% and 7%, respectively.

#### Closing Price and Moving Averages

![](/images/SBIN-closing-price-ma.png)

* All three moving averages showed a sustained upward trend from September 2025 onwards.
* At the end of the period, the ordering closing price > MA10 > MA20 > MA50 reflected higher prices over progressively shorter periods.

![](/images/LICI-closing-price-ma.png)

* The moving averages captured the recovery towards June–July and the subsequent downward trend.
* By January 2026, MA10 and MA20 remained below MA50, reflecting lower recent prices relative to the longer-term average.

![](/images/HAL-closing-price-ma.png)

* The moving averages rose during the March–June recovery, then declined following the mid-year price peak.
* MA10 tracked short-term fluctuations closely, while MA50 responded more gradually; the final closing price rose above all three averages.

![](/images/NTPC-closing-price-ma.png)

* The moving averages rose sharply during March–April, followed by relatively stable movement through much of the remaining year.
* In January 2026, the closing price and shorter moving averages moved above MA50, reflecting a recent recovery.

![](/images/ONGC-closing-price-ma.png)

* Repeated crossings between the closing price and moving averages reflected alternating periods of price increases and declines.
* The sharp January 2026 price increase exceeded all three averages, with MA50 showing the slowest response.

#### Daily Trading Volume

![](/images/daily-trading-volume.png)

* SBI maintained relatively consistent trading activity, with daily volumes commonly around 5–15 million shares; HAL generally recorded lower volumes, particularly during the second half of 2025.
* NTPC and LIC exhibited isolated volume spikes of approximately 100 million and 45 million shares, respectively, around June and late May 2025.
* ONGC showed substantial variation in trading volume, with a pronounced spike exceeding 70 million shares towards the end of January 2026.

#### Daily Percentage Returns

![](/images/daily-percentage-returns.png)

* Daily returns fluctuated around 0%, with most movements remaining within approximately −2% to +2%, particularly during the second half of 2025.
* HAL displayed frequent large positive and negative movements during January–May 2025, while SBI generally exhibited smaller fluctuations.
* Notable isolated movements included an approximately 8% increase in LIC during May 2025 and ONGC returns near −7% and +8% during March 2025 and January 2026, respectively.

#### Distribution of Daily Returns

![](/images/distribution-of-daily-returns.png)

* All five return distributions were concentrated near 0%, with most daily changes falling approximately between −2% and +2%.
* HAL exhibited a wider spread, indicating greater daily variability, while SBI and NTPC showed stronger concentration near zero.
* Sparse observations extended to approximately −7% and +8%, demonstrating occasional large price changes despite generally modest daily movements.

#### Closing Price Correlation

![](/images/closing-price-correlation.png)

* LIC and HAL exhibited the strongest positive closing-price correlation (0.81), indicating closely aligned price patterns during the observed period.
* SBI–HAL (0.41) and HAL–NTPC (0.39) showed moderate positive associations; most other pairs had weaker correlations.
* LIC–NTPC (0.03) and NTPC–ONGC (0.04) showed negligible linear association in closing-price levels.

#### Daily Return Correlation

![](/images/daily-return-correlation.png)

* All company pairs exhibited positive daily-return correlations between 0.36 and 0.51, indicating moderate co-movement during the observed period.
* SBI–NTPC (0.51) showed the strongest association, followed by SBI–LIC and LIC–NTPC, both at 0.48.
* HAL–NTPC (0.36) showed the weakest association; notably, LIC–HAL return correlation (0.40) was considerably lower than their closing-price correlation (0.81).

#### Historical Daily Risk Versus Return

![](/images/daily-risk-vs-return.png)

* SBI recorded the highest mean daily return (approximately 0.12%) and the lowest daily volatility (approximately 1.13%) during the observed period.
* HAL exhibited the highest volatility (approximately 1.97%), while its mean daily return was similar to ONGC’s at approximately 0.06%.
* NTPC showed a modest positive mean daily return (approximately 0.03%), whereas LIC was the only company with a negative mean daily return (approximately −0.02%).

## Preprocessing

* Monthly CSV files were consolidated separately for each company, sorted chronologically, and deduplicated by trading date.
* Dates and numerical fields were standardised, and missing values and invalid OHLCV records were checked.
* Initial records without complete moving-average features were excluded from modelling.
* Features and target prices were normalised using min–max scaling fitted exclusively on the training data to prevent test-data leakage.

## Modelling

A separate single-layer recurrent neural network (RNN) was trained for each company using PyTorch to predict the next trading day’s closing price. Each daily input contained four normalised features: the closing price and its 10-day, 20-day, and 50-day moving averages.

**Hyperparameters**

| Parameter | Setting |
|---|---|
| Recurrent layers | 1 |
| Hidden units | 64 |
| Input features | 4 |
| Sequence length | 60 trading days |
| Output | One predicted closing price |
| Hidden activation | Tanh |
| Output activation | Linear |
| Optimiser | Adam |
| Learning rate | 0.001 |
| Epochs | 100 |
| Batch size | 32 |
| Loss function | Mean squared error |
| Gradient clipping | Maximum gradient norm of 1.0 |
| Random seed | 42 |

**Window construction and input representation**

Each input sequence comprised 60 consecutive trading-day records. For a target day $t$, the model received observations from $t-60$ through $t-1$ and predicted the closing price on day $t$:

$$
    X_t=
    \begin{bmatrix}
    Close_{t-60} & MA10_{t-60} & MA20_{t-60} & MA50_{t-60}\\
    \vdots & \vdots & \vdots & \vdots\\
    Close_{t-1} & MA10_{t-1} & MA20_{t-1} & MA50_{t-1}
    \end{bmatrix},
    \qquad y_t=Close_t
$$

Each sequence therefore had shape $60\times4$. The window advanced by one trading day for each subsequent example. For instance, retained records 1–60 predicted record 61, while records 2–61 predicted record 62.

From 269 daily records, removing the initial 49 records without MA50 values left 220 complete feature records. Applying the 60-day window produced 160 supervised examples, divided chronologically into 128 training examples and 32 test examples using an 80:20 split.

**Processing within the RNN**

A full training batch had shape $32\times60\times4$, representing 32 sequences, 60 time steps per sequence, and four features per time step. At each step, the RNN received a $32\times4$ feature matrix and updated a $32\times64$ hidden-state matrix:

$$
    h_k=\tanh(x_kW_x+h_{k-1}W_h+b_h)
$$

The hidden state was initialised to zero for each sequence and carried information through its 60 time steps. The final hidden state was passed to a linear output layer:

$$
    \hat{y}=h_{60}W_y+b_y
$$

This generated one normalised closing-price prediction per sequence. Training batches preserved chronological order, and model parameters were updated to minimise MSE.

**Testing and evaluation**

Predictions were transformed back to the original price scale and evaluated using MSE, MAE, RMSE, and MAPE. Testing used rolling one-day-ahead predictions: each window included only observations available before its target day, including previously observed test-period prices.

## Results

The single-layer RNN was evaluated separately for each company on 32 test observations using rolling one-day-ahead predictions. Although the models captured several price patterns, the last-close baseline achieved lower errors across all four evaluation metrics for every company.

#### Training Performance

Training MSE decreased sharply during the initial epochs and generally stabilised at low values thereafter. Brief fluctuations occurred near the end of training for ONGC and NTPC. These curves demonstrate improved fitting of the training data; test performance was assessed independently.

![](/images/training-loss.png)

#### Test Metrics

Metrics were calculated after converting predictions back to the original price scale.

| Company | Method | MSE ($\text{INR}^2$) | MAE ($\text{INR}$) | RMSE ($\text{INR}$) | MAPE ($\%$) |
|---|---|---:|---:|---:|---:|
| SBI | RNN | 1,285.89 | 28.64 | 35.86 | 2.78 |
| SBI | Last close | 104.60 | 8.24 | 10.23 | 0.81 |
| LIC | RNN | 72.79 | 7.23 | 8.53 | 1.75 |
| LIC | Last close | 14.32 | 3.04 | 3.78 | 0.73 |
| HAL | RNN | 7,017.56 | 61.22 | 83.77 | 1.39 |
| HAL | Last close | 5,414.76 | 52.17 | 73.59 | 1.18 |
| NTPC | RNN | 39.43 | 4.51 | 6.28 | 1.32 |
| NTPC | Last close | 26.24 | 3.64 | 5.12 | 1.06 |
| ONGC | RNN | 47.51 | 4.53 | 6.89 | 1.81 |
| ONGC | Last close | 25.53 | 3.41 | 5.05 | 1.38 |

Among the RNN models, NTPC recorded the lowest MAPE (1.32%), followed by HAL (1.39%). SBI had the highest MAPE (2.78%) and the largest discrepancy relative to its baseline. HAL’s larger absolute errors partly reflect its higher share-price scale; MAPE provides a relative comparison across companies.

![](/images/rnn-vs-last-close.png)

#### Actual versus Predicted Prices

SBI: Predictions followed the broad upward direction but increasingly underestimated the January price increase. Towards the end of the test period, actual prices exceeded ₹1,070 while predictions remained near ₹1,000.

![](/images/SBIN-actual-vs-predicted.png)

LIC: The model captured the general downward movement but overestimated closing prices across much of the test period. The discrepancy became more pronounced during the January decline.

![](/images/LICI-actual-vs-predicted.png)

HAL: Predictions reproduced several broad fluctuations but responded late to abrupt changes, particularly the decline and subsequent recovery in late January.

![](/images/HAL-actual-vs-predicted.png)

NTPC: Predictions tracked much of the overall movement but underestimated the sharp increase around the beginning of January and the final price peak.

![](/images/NTPC-actual-vs-predicted.png)

ONGC: Predictions were relatively close during gradual movements but substantially underestimated the sharp price increase towards the end of January.

![](/images/ONGC-actual-vs-predicted.png)

The predictions generally exhibited smoother movements and delayed responses to abrupt price changes. Despite low training loss, the RNN did not improve upon the last-close baseline in this experiment. These findings are limited to the observed test period of 32 trading days per company.

## References

*Chaudhary, R. (2025, May 8). Advanced stock market Prediction using Long Short-Term Memory Networks: A comprehensive deep learning framework. arXiv.org. https://arxiv.org/abs/2505.05325*
