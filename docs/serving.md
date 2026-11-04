# Serving a classifier over HTTP

```bash
pip install "adaptive-classifier[serve]"
python -m adaptive_classifier.serving ./my_classifier --port 8000
```

`./my_classifier` is a directory written by `classifier.save(...)`. It is self-contained: it holds the
encoder (ONNX), the tokenizer and the learned state, so the server starts without reaching the Hugging
Face Hub. (Directories saved with `include_onnx=False`, or by an older version, fetch the base model
from the Hub when they start.)

## Endpoints

| Method and path | Body | Returns |
| - | - | - |
| `GET /health` | | `{"status": "ok", "classes": 3}` |
| `GET /info` | | classes, examples per class, calibration state, limits, whether updates are allowed |
| `POST /predict` | `{"text", "k"?, "abstain_below"?, "abstain_ood"?}` | `{"predictions": [{"label", "score"}], "abstained": false}` |
| `POST /predict_batch` | `{"texts": [...], "k"?, "ood"?, "ood_threshold"?}` | `{"predictions": [[...], ...]}`, plus `ood` and `ood_threshold` when `"ood": true` |
| `POST /predict_set` | `{"text", "alpha"?}` | conformal label set (needs `calibrate`) |
| `POST /ood` | `{"text", "threshold"?}` | `{"score", "threshold", "ood"}` |
| `POST /examples` | `{"texts": [...], "labels": [...]}` | add examples or new classes (updates only) |
| `POST /calibrate` | `{"texts": [...], "labels": [...]}` | fit confidence calibration (updates only) |
| `POST /forget` | `{"labels": [...]}` | remove classes (updates only) |
| `POST /remove_examples` | `{"texts": [...], "label"?}` | remove examples (updates only) |

Every list in a request holds at most `--max-batch` items (default 1000).

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
     -d '{"text": "where is my refund?", "k": 3}'
```

Bad input gets a 4xx with a message: 422 for a malformed request (empty text, wrong types, batch too
large) and 400 when the classifier rejects it (unknown label, calibration missing, ...). The interactive
API docs are at `/docs`.

## Updates

Learning at runtime is the point of the library, but it is off by default on a server:

```bash
export ADAPTIVE_CLASSIFIER_API_KEY=change-me
python -m adaptive_classifier.serving ./my_classifier --allow-updates --save-dir ./my_classifier
```

Update endpoints then require `Authorization: Bearer change-me`; prediction endpoints stay open.
`--save-dir` saves the classifier after every update, so a restart keeps what was learned.
Without an API key anyone who can reach the server can change the model, and the server warns about it.

### Saves are cheap and atomic

The encoder is never trained; only the memory, the neural head and the calibration change. A save
therefore exports (and quantizes) the encoder to ONNX only when `--save-dir` does not already hold an
export of the same base model (judged by `model_name` in its `config.json`). Every later save writes
just `model.safetensors` (prototypes, head, held example embeddings), `examples.json` (their texts)
and `config.json` (label maps, calibration, settings). The tokenizer files are also left alone once
they are there. From Python, `classifier.save(path, reuse_onnx=False)` forces a fresh export.

Each of those three files is written to a temporary file in the same directory and moved into place
with `os.replace`, so a crash or a reader never sees a half-written file; `config.json` is replaced
last. On Windows, replacing a file fails while another process holds it open without delete sharing
(a backup agent, a virus scanner, another process that loaded the model); the save retries for a few
seconds and then fails the request with the error. The classifier itself reads `model.safetensors`
into memory when it loads instead of memory-mapping it, so a server can save over the directory it
was started from. Saves are serialised, so two updates finishing together cannot interleave files.

The server saves **every held example** (`save(..., all_examples=True)`), not only the
`num_representative_examples` a plain `save` keeps, so a restart resumes with exactly the same
memory, prototypes and OOD radii.

### Calibration

`POST /calibrate` fits temperature scaling on labelled texts the model was *not* trained on (hold back
part of each LLM-labelled batch for it), saves when `--save-dir` is set, and returns:

```json
{"temperature": 1.4, "n": 200, "ece_before": 0.081, "ece_after": 0.024,
 "nll_before": 0.41, "nll_after": 0.33, "accuracy": 0.93,
 "calibrated_at": "2026-11-04T12:00:00+00:00"}
```

```bash
curl -s localhost:8000/calibrate -H "Authorization: Bearer $ADAPTIVE_CLASSIFIER_API_KEY" \
     -H 'content-type: application/json' \
     -d '{"texts": ["Fed raises rates ...", "Ten celebrity diets ..."], "labels": ["relevant", "off_topic"]}'
```

It needs at least 2 texts and 2 classes (20+ texts for a stable estimate). `/info` reports
`last_calibrated_at` (ISO-8601 UTC, `null` before the first calibration) and
`examples_since_calibration` (examples added through `/examples` since then); both are saved with the
classifier. Recalibrate when that count grows large relative to the window held in memory. Adding a new
class or forgetting one discards the calibration (`calibrated` turns false), because it described a
class set that no longer exists.

### Retention: `max_examples_per_class`

Each class holds at most `max_examples_per_class` examples (a classifier config option, default 1000;
override it at start-up with `--max-examples-per-class N`, which prunes at once if the saved memory is
larger). When a class goes over the limit it is pruned with a **recent + boundary** policy:

1. The newest `ceil(retention_recent_fraction * N)` examples are always kept (default fraction 0.5, so
   500 of 1000; set it with `--retention-recent-fraction` or the `retention_recent_fraction` config
   option).
2. The other slots go to the older examples **closest to another class's prototype** (Euclidean distance
   to the nearest other prototype; ties go to the older example). With only one class, those slots also
   go to the newest examples.

Kept examples stay in arrival order, so "newest" keeps its meaning at the next pruning.

Why this split: the gate's job is the boundary between `relevant` and `off_topic`, and the examples
that define that boundary are the borderline ones. The previous policy kept the examples closest to the
class mean, which discards exactly those, sharpens the head on easy cases only, and shrinks each
class's spread, so OOD radii shrank and ordinary borderline texts started to look out of
distribution. Pure newest-N (or reservoir sampling) keeps whatever happened to arrive recently and
lets rare hard cases age out. A news stream also drifts, so half the cap tracks what the LLM labelled
most recently; the other half preserves the hard cases that are still near the boundary, whenever
they arrived. 0.5 weights the two equally; raise it if the topic drifts fast, lower it if the boundary
is stable but subtle. With 100-1,000 labelled texts per cycle split over two classes and `N = 1000`,
the recent half covers roughly the last 1-10 cycles.

Everything follows the retained examples after pruning: the prototype is the mean of the held examples,
the search index is rebuilt, the OOD radius is measured on them, and the neural head is retrained on
them by the same update (the head is retrained on the whole memory after every `/examples` call;
`set_max_examples_per_class` retrains it when it evicts). Because boundary examples are kept, each
prototype leans slightly towards the boundary and each radius stays representative of the class's
real spread. A classifier loaded from a directory saved with only representative examples blends its
saved prototype (standing in for up to `N - new` older examples) with the examples added since
loading, until the class is first pruned; from then on the saved prototype and saved OOD radius are
dropped and only the held examples count. `training_history` keeps all-time counts. A failed update
restores the pruned examples along with everything else.

### Out-of-distribution scores in `/predict_batch`

Add `"ood": true` to get an OOD score for every text from the same encoder pass as the predictions:

```json
{"texts": ["Fed raises rates ...", "Ten celebrity diets ..."], "k": 2, "ood": true}
```

```json
{"predictions": [[{"label": "relevant", "score": 0.91}, {"label": "off_topic", "score": 0.09}],
                 [{"label": "off_topic", "score": 0.77}, {"label": "relevant", "score": 0.23}]],
 "ood": [{"ood_score": 0.84, "ood": false}, {"ood_score": 1.62, "ood": true}],
 "ood_threshold": 1.25}
```

`ood` is index-aligned with `predictions`. `ood_threshold` in the request overrides the configured
threshold for that call. Without `"ood": true` the response is exactly `{"predictions": [...]}` as before.
`ood_score` is `null` only when the classifier has no classes.

**What the score means.** For each class, the radius is the `ood_radius_percentile`-th percentile
(default 95) of the distances from that class's held training examples to its prototype. A text's
`ood_score` is its distance to the nearest prototype divided by that class's radius (minimum
`ood_min_radius`, 0.05). An `ood_score` of **1.0** means the text is exactly as far from the nearest
class centre as the 95th-percentile training example of that class: about 5% of a class's own
training examples score above 1.0. A text is flagged `"ood": true` when its score is above
`ood_threshold` (default **1.25**, i.e. 25% further out than that percentile). Previously the radius was
the single farthest training example, which with thousands of examples covers nearly every input;
`ood_radius_percentile = 100` restores that. Change it with `--ood-radius-percentile` or the
`ood_radius_percentile` config option, and the threshold with the `ood_threshold` config option.

## Running it as an LLM gate

A CPU-only pre-filter with two classes, `relevant` and `off_topic`, fed by an LLM that labels what the
gate passes on:

```bash
export ADAPTIVE_CLASSIFIER_API_KEY=change-me
python -m adaptive_classifier.serving ./gate --allow-updates --save-dir ./gate \
       --max-batch 1000 --max-examples-per-class 1000 --retention-recent-fraction 0.5
```

Each cycle: `POST /predict_batch` with up to 1,000 texts and `"ood": true`; send the texts that are
`relevant` or `ood` to the LLM; `POST /examples` with what it labelled; now and then `POST /calibrate`
with a held-out slice and watch `examples_since_calibration` in `/info`.

## Concurrency and scaling

Every classifier operation takes a lock, so one process can safely serve concurrent requests while
learning. That means requests are processed one at a time inside a process. For more throughput run
several **processes** (for example several containers behind a load balancer); with a read-only
server this scales cleanly. Each process has its own copy of the model, so updates only reach the
worker that received them: run a single process if you enable updates.

From Python, the same pieces are available without HTTP: `await classifier.apredict(...)`,
`apredict_batch` and `aadd_examples` run in a worker thread so they do not block an event loop, and
`create_app(classifier)` returns an ASGI app you can mount in an existing FastAPI project.

## Docker

```bash
docker build -f docker/Dockerfile -t adaptive-classifier .
docker run --rm -p 8000:8000 -v "$PWD/my_classifier:/model:ro" adaptive-classifier
```

The image runs as a non-root user, has a health check on `/health`, and uses CPU-only PyTorch. CI builds it and queries it (`scripts/docker_smoke.py`, workflow `docker.yml`) whenever the Dockerfile, `pyproject.toml` or the server change, and weekly.
