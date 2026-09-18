# Retail Demand Forecasting, trained on Kaggle and served on Kubernetes

Seven-day demand forecasts for 50,000 store-product series, trained as a single global LightGBM
model, registered in MLflow, and served from a FastAPI container on a kind cluster with request
batching, a load test and Prometheus metrics.

Every number below was measured. The commands that produce each one are in this file.

## Results

**Forecast accuracy**, official evaluation week 2024-06-26 to 2024-07-02, scored once after all
modelling choices were fixed (350,000 rows):

| Model | WAPE | Bias (WPE) | MAE | RMSE |
|---|---|---|---|---|
| **LightGBM, served configuration** | **0.3317** | -0.034 | 0.3957 | 0.6730 |
| LightGBM with same-day discount (not served, see below) | 0.3238 | -0.047 | 0.3863 | 0.6697 |
| Last-7-day average (best baseline) | 0.3612 | +0.003 | 0.4309 | 0.7228 |
| Chronos-2 zero-shot (pretrained, not served) | 0.3494 | -0.068 | 0.4168 | 0.7003 |
| Same weekday, previous 4 weeks | 0.3962 | -0.091 | 0.4726 | 0.8129 |
| Same day last week (seasonal naive) | 0.4183 | +0.003 | 0.4991 | 0.8358 |

The served model beats the best baseline by 8.2% relative WAPE, and the pretrained foundation
model by 5.1%.

**Serving**, in-cluster on kind, 60-second locust runs through `kubectl port-forward`, 0 failures
in every run:

| Run | Concurrent users | Batching | Throughput | p50 | p95 | p99 | Mean batch |
|---|---|---|---|---|---|---|---|
| Normal load | 10 | on | 82.2 rps | 100 ms | 250 ms | 390 ms | 8.2 |
| Overload | 50 | on | 81.4 rps | 570 ms | 990 ms | 1100 ms | 23.7 |
| Overload, batching disabled | 50 | off | 27.3 rps | 1700 ms | 3200 ms | 4500 ms | 1.0 |

Request batching is worth **3.0x throughput** and a **4x lower p99** at the same concurrency. The
mechanism is visible in the service's own metrics: per request, feature construction costs 233 ms
unbatched against 3.3 ms batched, and model inference 57 ms against 7.4 ms, because both are paid
once per batch rather than once per request.

Throughput is the same at 10 and 50 users, so the single-worker service saturates at roughly
82 rps. Past that point extra concurrency only adds queueing, which is what the 570 ms p50 in the
overload run represents. This is reported rather than tuned away; raising the ceiling is a matter
of replicas or workers and would not change what the numbers demonstrate.

## The dataset and why it was chosen

[FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) (Dingdong
Inc., CC BY 4.0, [arXiv:2505.16319](https://arxiv.org/abs/2505.16319)), pinned to revision
`08c1fab7f9257bc73679d415d65d644165d351d4` and checked by SHA-256 on every download.

50,000 store-product series, 898 stores, 865 products, 18 cities. 90 daily training days
(2024-03-28 to 2024-06-25) and a 7-day evaluation split (2024-06-26 to 2024-07-02). 4.5M training
rows, 350k evaluation rows, no nulls, a complete panel verified in notebook 1.

It was chosen over the familiar Kaggle competition datasets because its licence is explicit, its
data is real rather than simulated, and it carries hourly stockout annotations: 44.3% of training
rows have at least one out-of-stock hour, so recorded sales understate demand in a way the dataset
lets you measure. Store Item Demand is widely believed to be simulated and has no promotion flag;
Rossmann has no SKU dimension; the competition rules for both could not be confirmed.

## How leakage is prevented

Time-series data punishes random splits: a random split trains on days that come after the days it
is scored on. Every split here is by date, and the feature code enforces a stronger rule.

**Every history feature for target day `t` reads only days up to `t - 7`.** With a 7-day horizon,
one feature definition is therefore valid for all seven steps from a single forecast origin, no
prediction is ever fed back in, and the same code path serves training and inference. The cost is
that the nearest horizon days cannot use the most recent six days of history, and it shows: on
forecast day 2 the model is level with the baseline (0.3387 against 0.3381) while on days 4 to 7 it
is clearly ahead (0.3094 to 0.3624 against 0.3450 to 0.3976).

Same-day columns are split explicitly:

| Column | Treated as | Why |
|---|---|---|
| `holiday_flag` | known in advance | constant per date across all stores; a calendar |
| `activity_flag` | known in advance | planned promotion, matching the dataset paper |
| `discount` | **excluded from the served model** | could not be confirmed as set in advance |
| stockout hours | realized, never same-day | stock state is only observed after the day |
| weather | realized, excluded | actual weather is not a forecast; available only as a labelled oracle variant |

Discount was excluded after notebook 1 found that `discount = 1.0` appears on 65.7% of zero-sale
days against 47.7% of days with sales, which is what you would expect if the column were derived
from transactions. Including it improves evaluation WAPE to 0.3238. That difference is the cost of
not relying on an unverified assumption, and it is reported rather than quietly taken.

The leakage rules are enforced by tests, not by intention:

- 9 unit tests on a synthetic panel, including a positive control that fails when leakage is
  introduced deliberately.
- On the real data, masking or randomizing the target, stockout hours and weather for the
  validation and evaluation weeks leaves every feature on those days bit-identical.
- Serving parity: features rebuilt from only the last 34 days of history match the full-history
  features exactly, and the saved model file on that 34-day snapshot reproduces all 350,000
  evaluation predictions exactly.

## The model

A single global LightGBM model across all series. Objective chosen on a validation week (the last
7 training days) among squared error (0.3342 WAPE), absolute error (0.3333) and Tweedie (0.3361);
absolute error won and was used for the served model. Actual-weather variant (0.3301) was measured
and discarded as unavailable at forecast time.

- 381 boosting rounds, 33 features, trained on target days 2024-05-01 to 2024-06-25 (the first day
  with a complete 34-day feature history onward).
- Features: lags 7 to 14, 21 and 28; rolling means and standard deviations over windows ending at
  `t-7`; same-weekday mean over 4 weeks; zero-sale share; past stockout hours; promotion history;
  calendar; store, product, category and city identifiers.
- Predictions clipped at zero, which is part of the registered model rather than a serving detail.

Largest feature gains: 14-day rolling mean 29.8%, `product_id` 15.4%, `store_id` 14.8%, 7-day mean
9.6%, 28-day mean 8.8%.

## Architecture

```
Kaggle notebook (CPU)            local machine                       kind cluster
-------------------------        ------------------------------      ---------------------------
01 data, features, baselines  →  mlops/register_model.py          →  init container seeds a
02 LightGBM, eval, exports       logs run + registers model          writable copy of the store
   model file, model card,       mlops/export_model_store.py      →  FastAPI pod resolves
   parity sample, 34-day         rewrites artifact paths for         models:/retail-demand-lgbm/1
   history snapshot              the container                       Prometheus scrapes /metrics
```

The service resolves the model through the MLflow registry rather than opening the model file, so
what runs in the pod is the registered version. The registry database travels with the image, its
artifact locations rewritten from host paths to the container path.

Endpoints: `/predict`, `/model`, `/healthz`, `/readyz`, `/metrics`. The forecast request names a
store, product and date, plus the known-in-advance columns; the service holds the 34-day history
snapshot itself.

## Verification trail

Each step is checked against the step before it, by hash or by exact prediction match:

| Check | Result |
|---|---|
| Kaggle rerun of notebooks 1 and 2 vs a sandbox rerun | 675 and 1,045 metric values match to rounding; module hashes identical |
| Model file vs the hash in its model card | match |
| MLflow registry reload (`mlflow.pyfunc.load_model`) vs Kaggle predictions | 2,000 of 2,000 identical |
| Exported container store vs Kaggle predictions | 2,000 of 2,000 identical |
| Live pod vs sandbox for the same request | 0.2502280936944047 both |
| `forecast_model_info` in Prometheus | model hash `eacdebeed7b701f9`, version 1 |

## Chronos-2: measured, not served

[Chronos-2](https://huggingface.co/amazon/chronos-2) (Amazon, Apache 2.0, 120M parameters) is a
pretrained forecasting model that needs no training on this dataset. Notebook 3 runs it zero-shot
over all 50,000 series on the same evaluation week, with the same information the served model
gets: past stockout hours and discount, known-in-advance promotion and holiday flags, no same-day
discount, no weather. 219.8 seconds on a Kaggle T4.

**Accuracy.** Chronos-2 reaches 0.3494 WAPE: better than every classical baseline, 5.1% worse than
the trained LightGBM model. It wins forecast day 2 (0.3344 against 0.3387) and loses the other six
days, most clearly day 4 (0.3390 against 0.3094). Its bias is twice the LightGBM model's
(-0.068 against -0.034). Its 80% prediction interval covered 74.0% of actuals, so the intervals are
overconfident, which matches published findings on pretrained forecasters.

**Cost.** This is what decides it:

| | Chronos-2 | Served LightGBM |
|---|---|---|
| Throughput, GPU (T4, batch 512) | 229.1 series/s | not applicable |
| Throughput, CPU | 19.0 series/s (52.6 ms/series) | ~5 ms of model time per request |
| Serving hardware | GPU for usable throughput | one CPU pod, 82 rps, p50 100 ms |
| Context required per request | 90 days | 34 days |

On the CPU the service actually runs on, Chronos-2 is roughly 10x slower per series while being
less accurate, and it needs nearly three times the history per request. A GPU closes the throughput
gap and adds cost. So the decision is straightforward on this workload: LightGBM is served, and
Chronos-2's value here is as a reference point that the trained model has to beat, which it does.

**Contamination cannot be ruled out.** Chronos-2's pretraining corpus is not fully disclosed, and
this dataset was published before the model. If the dataset or correlated series appeared in
pretraining, the zero-shot score flatters the model. Treat the comparison as indicative.

## Known limitations

- **Censored actuals.** Recorded sales on stockout days understate demand, so a model that matches
  them learns that understatement. The bias is reported, not corrected: on full-day-stockout rows
  the served model's MAE is 0.9171, and WAPE there is meaningless because recorded sales are near
  zero. Demand recovery (Tobit or inverse-Mills) is a deliberate non-goal for this version.
- **Near horizons.** The lag ≥ 7 rule costs accuracy on forecast days 1 to 3, where the simple
  baseline is competitive. Per-horizon models would fix it and are out of scope.
- **Identifier reliance.** `store_id` and `product_id` take about 30% of total gain between them,
  with 66,000 splits. That may be series memorization; a holdout of unseen stores or products would
  settle it. Not investigated.
- **Normalized units.** The publisher normalized sales, so errors are not in physical units and
  MAE is not directly interpretable as items.
- **Ninety days of history.** No yearly seasonality is learnable, and no year-over-year features
  exist.
- **Single worker.** The service saturates at about 82 rps; the load test measures one replica with
  one uvicorn worker.
- **Load generated through a port-forward.** Client and server percentiles agree within about
  50 ms, so this did not distort the numbers, but a NodePort run would remove the caveat.
- **Pretraining contamination is unknowable.** The Chronos-2 comparison cannot confirm that this
  dataset was absent from that model's pretraining corpus.

## Reproducing it

Training, on Kaggle (Internet on, CPU; each notebook is self-contained and downloads the pinned
dataset itself):

1. `notebooks/01_data_features_baselines.ipynb` — validation, features, leakage tests, baselines.
   38 checks, about 3 minutes.
2. `notebooks/02_lightgbm_training_evaluation.ipynb` — objective comparison, ablations, final
   evaluation, exports. 17 checks, about 30 minutes.
3. `notebooks/03_chronos2_offline_comparison.ipynb` — Chronos-2 zero-shot comparison and throughput
   benchmark. Needs a GPU accelerator; about 6 minutes on a T4.

Download the notebook 2 outputs, then locally:

```powershell
python mlops\register_model.py --artifacts-dir "Output Files\Notebook 2" `
  --tracking-uri "sqlite:///<project path>/mlflow.db" --artifact-root "<project path>\mlartifacts"
python mlops\export_model_store.py --summary "Output Files\Notebook 2\mlflow_registration_summary.json" `
  --artifacts-dir "Output Files\Notebook 2" --out service\model_store
copy "Output Files\Notebook 2\nb2_serving_history.parquet" service\data\serving_history.parquet

docker build -f service/Dockerfile -t retail-demand-api:1.0.0 .
kind create cluster --name demand-forecast --image kindest/node:v1.34.3
kind load docker-image retail-demand-api:1.0.0 --name demand-forecast
kubectl apply -f k8s/
kubectl -n demand-forecast rollout status deploy/forecast-api --timeout=240s
```

Then, with `kubectl -n demand-forecast port-forward svc/forecast-api 8000:8000` running:

```powershell
python loadtest\run_loadtest.py --host http://127.0.0.1:8000 --label k8s-batching-on `
  --users 10 --spawn-rate 10 --run-time 60s --history "Output Files\Notebook 2\nb2_serving_history.parquet"
```

Toggle batching with `kubectl -n demand-forecast set env deploy/forecast-api BATCHING_ENABLED=false`
and remove the override with a trailing `-`.

## Repository layout

```
notebooks/      Kaggle notebooks 1-3 (assembled by build_notebook*.py from src/ and tests/)
src/            demand_features.py, demand_eval.py, demand_model.py - shared by notebooks and service
tests/          14 tests: feature leakage, window maths, model contract
mlops/          register_model.py (MLflow run + registry), export_model_store.py (container store)
service/        FastAPI app, Dockerfile, 8 service tests
k8s/            namespace, RBAC, configmap, deployment, service, Prometheus config and deployment
loadtest/       locustfile and a runner that records percentiles plus server-side metrics
reference_runs/ sandbox executions kept for comparison against the Kaggle and cluster runs
```

The notebooks embed `src/` and `tests/` verbatim through `build_notebook*.py`, so notebook code and
serving code cannot drift apart.

## Deployment notes worth knowing

Three failures hit during deployment, each fixed rather than worked around:

1. `kindest/node:v1.35.0` would not start a control plane on this machine; the node image is pinned
   to v1.34.3.
2. MLflow writes a `registered_model_meta` file into the model directory on every load, which fails
   on a read-only root filesystem. An init container seeds a writable copy, and the read-only root
   filesystem is kept.
3. `python:3.12-slim` does not ship `libgomp1`, which LightGBM needs. The Dockerfile installs it and
   now runs a build-time smoke check as the runtime user, so a missing library fails the build
   instead of crash-looping the pod.
