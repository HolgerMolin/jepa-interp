# V-JEPA physics experiments

For a shared checkout, start by reading the rendered notebook or
[presentation PDF](presentation/vjepa_physics_draft.pdf); both include saved outputs.
The repository includes experiment code, reports, frozen splits, dataset metadata,
and the saved prediction artifacts used by the presentation analyses.

To run the CPU tests in an environment with `requirements.txt` installed:

```bash
python -m unittest discover -s misc -t . -p 'test_*.py'
```

Sections 1 and 6 can reproduce the saved predictor analysis without videos,
feature caches, or a GPU. Full fitting and GPU replay require the separately
stored videos, scratch feature caches, and pinned checkpoint. Adjust the notebook's
`SCRATCH` and `CHECKPOINT` paths for your machine. The shell runners below retain
the original FarmShare allocation paths. `.gitignore` excludes videos, model
weights, large intermediate fits, caches, logs, and superseded drafts; no excluded
experiment files have been deleted from the original workspace.

Open **[experiments.ipynb](experiments.ipynb)** and read/run it from top to bottom.
The notebook follows the [presentation](presentation/vjepa_physics_draft.pdf):
layer-wise probes, iterative nullspaces, multi-probe steering, offset-preserving
centroid splines, then velocity interventions through the frozen predictor.
Short explanations separate independent-readout error reduction from downstream
velocity response. The original pooled calculations and output formats remain
unchanged; the earlier PCA-replacement evaluation is retained in Appendix C.

- **[utils.py](utils.py)** contains the small data/model, extraction, ridge, and metric helpers.
- **[misc/](misc/README.md)** contains the standalone experiment scripts, tests, and shell runners.
- **[requirements.txt](requirements.txt)** lists the runtime dependencies.
- **[results/](results/)** retains the original splits, probes, plots, reports, and audit.
- **[misc/archive/](misc/archive/)** retains historical extraction, probing, and audit code for reference.
- **[vjepa-physics-takehome-4E00/](vjepa-physics-takehome-4E00/)** is the untouched supplied repository.

## Run

Run legacy Python entry points from the project root as `python -m misc.MODULE`
and shell runners as `bash misc/SCRIPT.sh`. Their numerical routines and data/output
locations are unchanged. Source-hash guards remain enabled; new fits should use
fresh output directories when a frozen protocol detects the relocated source.

Open the notebook from this directory and select the existing interpreter:
`/scratch/users/hmolin/jepa-interp/.venv/bin/python`.
For another environment, install `requirements.txt` and adjust the notebook's paths.

Cached analysis needs a CPU compute allocation. On FarmShare, request one with
`salloc -p normal -c 4 --mem=16G -t 00:30:00` and start the notebook kernel on the
allocated compute node (for example through a Jupyter session started with `srun`).
The notebook can also be executed headlessly on that allocation:

```bash
srun /scratch/users/hmolin/jepa-interp/.venv/bin/python - <<'PY'
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('experiments.ipynb', as_version=4)
NotebookClient(nb, timeout=1800).execute()
nbformat.write(nb, 'experiments.ipynb')
PY
```

Features are one `.npz` per dataset under
`/scratch/users/hmolin/jepa-interp/pooled_features/`, each containing
`features[video, layer, channel]` and matching `ids`.
The checkpoint is under `/scratch/users/hmolin/jepa-interp/checkpoints/`;
its pinned revision and checksum are in [checkpoint_manifest.json](checkpoint_manifest.json).

Fresh extraction is explicit: obtain a GPU allocation, set `RECOMPUTE_FEATURES = True`,
and run the extraction cell. A changed model/preprocessing protocol must use a new
cache directory. Existing caches are not automatically invalidated or resumed.

The notebook reuses frozen A/B/C/D splits. The original pooled fitting/evaluation
cells overwrite their generated files in `results/nullspace/`, `results/steering/`,
and `results/spline/`, as before. New presentation-analysis cells read saved artifacts
and keep their calculations in memory. They check fixed-target spline predictions,
path errors, predictor gains, and bootstrap intervals against the recorded results.

For the predictor analysis alone, run Section 1 (both setup cells), then Section 6.
No feature cache or GPU is needed for those saved-result calculations. The optional
`REPLAY_PREDICTOR` cell defaults to `False`; enabling it on a GPU allocation checks
two clips through the original frozen model using the saved token operators and spline.
The full experiment runners and correctness tests remain available below.

The [notebook validation record](results/notebook_presentation_checks.json) records
the saved-artifact comparisons, CPU edit checks, and test results. All 21 original
computational cells were retained verbatim; fresh GPU inference and a full rerun
of the original fits were not part of this notebook update.

## Interactive Jupyter on the current GPU allocation

From the FarmShare login shell (allocation 1903060, node oat-06):

```bash
srun --jobid=1903060 --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 \
  /scratch/users/hmolin/jepa-interp/.venv/bin/python -m jupyterlab \
  --no-browser --ip=127.0.0.1 --port=8888 --ServerApp.port_retries=0 \
  --ServerApp.root_dir=/home/users/hmolin/jepa-interp
```

In a separate terminal on your laptop:

```bash
ssh -N -J hmolin@login.farmshare.stanford.edu \
  -L 8888:127.0.0.1:8888 hmolin@oat-06
```

Open the localhost URL printed by Jupyter (including its token), then open
`experiments.ipynb` and choose Python 3. Keep both terminals open.
These allocation/node values apply to the current session; update them after
an allocation ends. Leave `RECOMPUTE_FEATURES = False` for cached analysis.

## Iterative nullspace probing

Section 3 uses block 19 and removes at most 256 independent dimensions per variable.
It repeatedly fits a fresh ridge probe on A, selects regularization on B, and removes
its readout row space. Validation determines stopping; D is scored only after all
probe sequences are fixed. C is reserved here for the independent steering evaluators in Section 4.

The notebook saves the protocol, validation decisions, scores, plot, and learned
bases/readouts under `results/nullspace/`. Running this section again overwrites
those outputs. The cap is dimensions, not iterations: direction can remove two
per probe. Projection uses an orthonormal basis rather than a dense matrix.

## Pooled video decoder

`misc/train_decoder.py` trains a small convolutional decoder from raw block-19 pooled
features to all 16 RGB frames at 64×64. It uses a fresh, seeded 80/20 split within
each dataset, independent of A/B/C/D, and fits feature normalization on training
clips only. Videos are downsampled once into roughly 900 MB of RAM. Run on a GPU
compute allocation using the existing environment:

```bash
/scratch/users/hmolin/jepa-interp/.venv/bin/python -m misc.train_decoder
```

The defaults are 100 epochs, batch size 32, AdamW, and a pixel MSE loss with 20×
weight on bright disk pixels. `--epochs` changes the training duration; `--output`
chooses a fresh output directory (default `results/decoder`). Existing output
directories are rejected to preserve previous runs. There is no resume system.
Outputs include `decoder.pt`, the saved split/settings, per-epoch train/test
losses, a loss plot, and fixed test reconstruction examples. Training losses
are measured during updates; test losses use the end-of-epoch model. The final
epoch is retained without selecting on test loss. If test results guide repeated
architecture changes, treat that split as development data.

For later steering, load the decoder and pass **raw**, possibly edited, pooled
features. Training normalization is stored inside the model:

```python
import torch
from misc.train_decoder import Decoder

saved = torch.load('results/decoder/decoder.pt', map_location='cpu', weights_only=True)
decoder = Decoder()
decoder.load_state_dict(saved['model'])
decoder.eval()
with torch.no_grad():
    reconstructed = decoder(raw_features.float())  # [batch, 16, 3, 64, 64], RGB in [0, 1]
```

The decoder reconstructs the full observed clip; it does not forecast unseen
frames. Inspect whether disks and trajectories survive reconstruction, since
background pixels dominate unweighted MSE. This decoder split deliberately
overlaps the probing partitions; report decoder and steering training membership
when interpreting later intervention results.

## Full-token layer sweep

`misc/probe_full_tokens.py` fits direction, speed, and acceleration ridge probes at
all 24 encoder blocks using **every token and channel**: 2,048×1,024 features per
clip. Token positions have independent coefficients; there is no spatial or
temporal pooling, token subsampling, dimensionality reduction, or shared-token
probe. Direction still predicts sine/cosine; scalar labels retain their original
units. These remain full-clip decoding probes, not prefix-only forecasting probes.

The original A/B/C/D roles are retained. A fits the readout and intercept; B
selects alpha and layers; C is unused; D is scored after all layer choices are
saved. The original alpha grid is extended upward to 1e8 for the larger feature
space. The solver uses exact dual ridge in float64, with feature-chunked Gram
accumulation, and recovers explicit token-shaped coefficients. There is no
feature scaling. Validation scores, train scores, all alpha candidates, and
grid-edge selections are recorded.

Fresh float32 inference processes one task and block at a time, updating a
roughly 10 GiB host activation buffer in place. It requires one GPU and 32 GiB
host RAM; GPU inference batches default to two clips. Pooled means for three
clips at every block are checked against the existing caches. No full-token
activation cache is written; the saved float64 readouts occupy about 1.6 GB.

From a terminal with Slurm access, launch in the existing allocation (replace
the job ID when needed):

```bash
nohup bash misc/run_full_token_probes.sh 1903060 > results/full-token-probes.log 2>&1 &
```

The runner writes to a fresh `results/full_token_probes/` directory and rejects
an existing directory. Outputs are `protocol.json`, incremental `validation.json`,
`selection.json`, final `scores.json`, and `layerwise.png` comparing full-token
and historical pooled test curves. Each `probe_TASK_layerNN.npz` contains
`coef[output, token, channel]`, `intercept[output]`, split IDs, and test predictions.
Read a raw activation tensor with `np.einsum('btc,otc->bo', tokens, coef) + intercept`.
Tokens are ordered by time, height, then width: the grid is 8×16×16. No final
encoder LayerNorm is applied to the probed block outputs.

CPU correctness checks:

```bash
/scratch/users/hmolin/jepa-interp/.venv/bin/python -m unittest -v misc.test_full_token_probes
```

## Independent probes for each token

`misc/probe_each_token.py` fits a separate ridge probe for each of the 2,048 token
positions at every one of 24 blocks, for each of the three original tasks:
147,456 probes in total. Each probe has 1,024 input channels; no weights are
shared across tokens and tokens are not treated as independent training videos.
Direction predicts sine/cosine, while speed and acceleration predict their
original scalar magnitudes. These are full-clip readouts, not forecasting probes.

The original methodology is retained: fit on A, center using A only, fit an
intercept without feature scaling, choose each token's alpha by B MAE from the
original nine-value grid (1e-4 through 1e4), reserve C, and score D. All token
and layer scores are saved; no token or layer is selected for editing. The logged
fraction with B R² >= 0.5 is descriptive. Subsequent editing thresholds must be
chosen from validation data, not test scores. Test intervals use the original
500 clip-bootstrap resamples, shared across token positions.

The exact ridge solution uses batched float64 eigensolves on a GPU, with one
factorization reused across alphas. The runner retains one block's activations
in host RAM and processes 16 independent probes per GPU batch. A job requests
one GPU, four CPU cores, 32 GB RAM, and 24 hours. It reports measured time and
remaining time per layer; the full sweep is substantially larger than pooled
probing. Final readouts, predictions, and scores take approximately 2–3 GB.

Submit a new one-GPU allocation from a terminal with Slurm access:

```bash
sbatch misc/run_token_probes.sbatch
```

To use one GPU within an existing allocation instead (substitute its job ID):

```bash
nohup srun --jobid=1903060 --overlap --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=32G --gres=gpu:1 \
  /bin/bash misc/run_token_probes.sbatch > results/token-probes-existing.log 2>&1 &
```

Outputs go to `results/token_probes/`. Each `TASK_layerNN.npz` contains:

- `coef[token, output, channel]` and `intercept[token, output]` for raw activations;
- token-specific `alpha`, grid-edge flags, and the full validation grid;
- train, validation, and test R²/MAE, plus per-token test confidence intervals;
- test predictions and IDs, training/validation IDs, and readout singular values;
- layer number, token-grid shape, extraction-equivalence checks, and runtime.

Token order is time, height, width, with grid 8×16×16. For token index `i`, the
prediction is `tokens[:, i] @ coef[i].T + intercept[i]`. `summary.json` gives
per-layer quantiles across tokens, and `layerwise.png` compares those summaries
with the historical pooled probes. Shaded plot bands show variability across
tokens, not confidence intervals.

Each layer file is written atomically. To resume an interrupted run:

```bash
sbatch misc/run_token_probes.sbatch --resume
```

Completed fits are reused; encoder activations are recomputed through earlier
blocks. Resume requires matching protocol and code hashes. A fresh run rejects
an existing output directory; `--output` selects another directory. Set
`--token-batch-size` or `--batch-size` for smaller GPU batches if necessary.

Correctness checks compare each token independently with the existing ridge
solver, verify video-split isolation and bootstrap equivalence, and check
layer-by-layer inference:

```bash
/scratch/users/hmolin/jepa-interp/.venv/bin/python -m unittest -v misc.test_each_token misc.test_full_token_probes
```

## Velocity edits through the pretrained predictor

The executed prefix-only experiment is documented in
[results/velocity_causal_v1/REPORT.md](results/velocity_causal_v1/REPORT.md).
It uses the first eight frames of all 1,536 constant-velocity speed clips, fits
velocity and position probes on each of the 1,024 layer-19 prefix tokens, edits
velocity while preserving the construction position readouts, and runs encoder
blocks 20–24, final LayerNorm, and the frozen pretrained predictor. Future frames
never enter the context encoder. A real-model leakage check replaces them with
noise and verifies unchanged activations and predictions.

`misc/velocity_causal.py` extracts features and fits C-only position evaluators;
`misc/velocity_intervention.py` fits A/B edit probes and evaluates fixed interventions.
The primary position decoder is shared across future horizons. Random controls
match each token's edit norm and preserve its construction position readouts.
`misc/summarize_velocity_causal.py` writes the report and paired-control comparisons.
Large features live in `/scratch/users/hmolin/jepa-interp/velocity_causal_v1/`.

The first run shows a small, directionally consistent effect: a +0.5 m/s edit
shifts the farthest predicted position by 0.34 pixels, versus 0.005 pixels for
the matched random control and 5 pixels for the physical expectation. The
velocity-like slope change is about 3.3% of the requested change. This measures
the frozen encoder–predictor–readout pipeline; a position decoder trained on
real future encoder tokens transfers poorly to predictor outputs, limiting
independent validation of counterfactual positions.

For a fresh run, use explicit new output/cache directories. These commands run
inside a GPU allocation with four CPU cores and 32 GB host RAM:

```bash
python -m misc.velocity_causal extract --output RESULTS --cache SCRATCH_CACHE
python -m misc.velocity_causal baseline --output RESULTS --cache SCRATCH_CACHE
python -m misc.velocity_intervention fit --output RESULTS --cache SCRATCH_CACHE
python -m misc.velocity_intervention intervene --output RESULTS --cache SCRATCH_CACHE
python -m misc.summarize_velocity_causal --output RESULTS
```

Use the interpreter documented above. Extraction rejects existing directories;
later stages use the saved artifacts. `fit` requires the predefined B baseline
gate to pass. The original run used allocation 1887909 on oat-02 via
`misc/run_velocity_causal.sh` and `misc/run_velocity_intervention.sh`; those runners retain
the original paths and are not fresh-run templates. Run the focused checks with
`python -m unittest -v misc.test_velocity_causal` on a compute allocation.

## Iterated per-token velocity probes

The follow-up in `misc/iterated_velocity.py` fits a separate orthogonal velocity-probe
sequence for each prefix token at layer 19, using the cached activations from
the predictor experiment. Each token's existing position readout is protected.
Only probes with **both** velocity components' validation R² at least 0.5 are
eligible for steering; sequence termination uses a separate sustained
near-baseline criterion or a cap of 256 removed dimensions per token.

`misc/evaluate_iterated_velocity.py` assembles simultaneous edits, evaluates increasing
maximum eligible-probe counts, compares position-preserving random controls,
runs matched-rank erasures, and audits velocity remaining in all tokens jointly.
The frozen C-trained position evaluators from the first run are reused. The
existing D set has already been inspected, so this is a follow-up experiment.

The default result directory is `results/iterated_velocity_v1/`; large sequence
weights and bases are under `/scratch/users/hmolin/jepa-interp/iterated_velocity_v1/`.
On a GPU allocation with the existing interpreter, four CPU cores, and 32 GB RAM:

```bash
python -m unittest -v misc.test_iterated_velocity
python -m misc.iterated_velocity fit
python -m misc.evaluate_iterated_velocity assemble
python -m misc.evaluate_iterated_velocity evaluate
python -m misc.evaluate_iterated_velocity audit
python -m misc.report_iterated_velocity
```

Fit chunks are atomic and reusable if the source hash matches. Evaluation saves
each condition independently. The run scripts use the default paths; explicit
`--output` and `--work` arguments select a fresh experiment directory. The final
report is `results/iterated_velocity_v1/REPORT.md`, with steering, erasure,
token-count maps, and joint-audit plots alongside it.

The completed run increased the +0.50 m/s velocity slope gain from 3.45% with
one probe to 17.91% with all eligible probes (median 3, range 1–23 per token).
Matched random steering stayed near zero. Terminal erasure left substantial
joint-token velocity decodability (D R² 0.89/0.86) and caused 14.85 px of
context-position drift, limiting selective causal interpretation of its damage.
See the [full follow-up report](results/iterated_velocity_v1/REPORT.md).

## Conditional token spline steering through the predictor

`misc/spline_velocity.py` fits speed-dependent edits to all prefix-only layer-19
tokens, conditioning on direction and boundary position. A shared natural cubic
speed / Fourier direction / quadratic position basis has separate coefficients
for each token/channel. A fits ridge coefficients; B selects complexity and
regularization using activation reconstruction error. Edits add the conditional
curve difference while preserving the existing position-probe rows.

The run compares ±0.25 and ±0.50 m/s edits with the saved iterated linear method
and per-token norm-matched random controls through the unchanged predictor and
position readouts. No downstream result selects or rescales the spline fit.

The completed fit did not improve on linear steering: at +0.50 m/s it gave
3.84% velocity slope gain versus 17.91%, with 8.29× the edit norm and 5.88× the
context-position drift. At −0.50 m/s gains were similar (18.33% versus 18.11%),
but the spline still caused substantially more drift. These findings concern
this conditional spline fit and the frozen predictor–readout pipeline.

See [the spline intervention report](results/spline_velocity_v1/REPORT.md).
`results/spline_velocity_v1/` contains the frozen protocol, model selection,
per-example predictions, metrics, and plots. Full coefficients are in
`/scratch/users/hmolin/jepa-interp/spline_velocity_v1/`.

On an allocated GPU with four CPU cores and 32 GB RAM, run
`bash misc/run_spline_velocity.sh`. It runs numerical tests, fits, evaluates, and
generates the report. Use fresh output/work directories for a changed protocol.

`misc/diagnose_spline_probes.py` evaluates those same edits with the 4,274 frozen,
B-eligible iterated velocity probes. For +0.50 m/s, the first probe reports
29.0% of the requested layer-19 velocity change, subsequent probes average 2.0%,
and all probes average 10.0% with equal token weighting. For −0.50 m/s those
figures are 33.6%, 10.4%, and 17.3%. The spline therefore does not impose the
requested change consistently across the discovered velocity readouts, even
before the predictor. These are construction-probe diagnostics, not independent
causal measurements. See the [probe diagnostic report](results/spline_velocity_v1/probe_diagnostic/REPORT.md).

## Original centroid spline recipe through the predictor

`misc/centroid_velocity.py` restores Section 5's original helpers unchanged:
A-fitted 64D pooled PCA, speed-bin centroids, exact natural cubic interpolation,
B selection among 8/12/16 anchors, and replacement of pooled PCA coordinates.
It refits on prefix-only layer-19 tokens and adds the same pooled change to each
token, the minimum-norm lift that exactly achieves that mean change. No
conditional regression or additional position projection is introduced.

The original pooled advantage reproduces: spline interior target MAE is
0.107 m/s versus 0.166 for endpoint chords. The speed probe reports 93–98% of
the requested speed change. Through the predictor, however, movement from the
source-speed centroid to the target gives only 5–6% of the physical velocity
response, versus approximately 18% for iterated linear steering.

Centroid replacement at the original speed is non-identity: it shifts context
position by 2.26 px and slows decoded motion by about 0.184 m/s. Both the total
effect from the untouched representation and the source-to-target effect are
reported; for positive speed requests the total motion response is negative.
These conclusions apply to this prefix-only refit and uniform token mapping.

See [the centroid transfer report](results/centroid_velocity_v1/REPORT.md).
Run `bash misc/run_centroid_velocity.sh` inside a GPU allocation to execute tests,
fit, evaluate, and report. Outputs are under `results/centroid_velocity_v1/`;
prefix-pooled features are cached under
`/scratch/users/hmolin/jepa-interp/centroid_velocity_v1/`.

## Per-token offset-preserving centroid follow-up

`misc/token_centroid_velocity.py` implements a separate empirical speed-centroid
spline at each of the 1,024 prefix tokens. Each token has its own A-fitted
64-dimensional PCA and B-selected 8/12/16-knot natural cubic spline. The edit
is `h_t + f_t(target_speed) - f_t(source_speed)`, retaining the source residual.
This is distinct from both the pooled/broadcast centroid experiment and the
earlier conditional regression spline.

The primary variant projects each selected curve off the same frozen position
readout directions protected by linear steering. It preserves the residual
relative to this projected curve. The unprojected curve is also evaluated.
Both use per-token norm-matched random controls, identical B/D clips and speed
requests, and unchanged predictor/position evaluators. Saved one-probe and
all-eligible-probe linear results are compared using paired clip bootstraps.
No downstream outcome selects the curves or the primary variant. This matches
token mapping and position constraints, but one speed-only spline still differs
from an iterated sequence of direction-aware linear velocity probes.

Implementation is available; a completed run is required before citing results.
Run on an existing GPU allocation (replace the allocation ID if it has ended):

```bash
srun --jobid=1887909 --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 \
  bash misc/run_token_centroid_velocity.sh
```

Outputs default to `results/token_centroid_velocity_v1/`, including per-token
models, frozen protocol/selection, per-clip predictions, paired comparisons,
and `REPORT.md`. `--output PATH` selects a fresh experiment directory. An
interrupted run can resume if source/input hashes are unchanged. Numerical
checks can be run separately with `python -m unittest -v misc.test_token_centroid_velocity`.

## Iterated centroid-spline residualization

`misc/iterated_centroid_velocity.py` refits PCA and centroid splines on residuals,
subtracting each curve's A-centered speed-dependent contribution. A fits each
curve; B chooses knots and the best sequence prefix. The fixed rule stops after
three rounds without a meaningful validation improvement, capped at 16 rounds.
Steering always adds cumulative source-to-target differences to the untouched
tokens; it never performs initial centroid replacement.

The run stopped after four attempted rounds and selected one. At +0.50 m/s,
predictor velocity response was 5.39% for one curve and 5.35% for four, with
context drift increasing from 0.298 to 0.339 px. Fresh linear speed-probe D R²
fell from 0.985 to approximately zero after the first subtraction, but an RBF
probe retained R² 0.908 after one round and 0.905 after four. The transformation
therefore leaves nonlinearly readable speed information despite removing most
linear decodability. Residuals use known speed labels; this is a supervised
diagnostic rather than a label-free erasure guarantee.

See [the iterative spline report](results/iterated_centroid_velocity_v1/REPORT.md).
Run `bash misc/run_iterated_centroid_velocity.sh` on a GPU allocation for tests,
sequence fitting, predictor evaluation, linear/RBF residual audits, and plots.
The saved output includes unsuccessful attempted prefixes for transparency;
only B reconstruction selects a prefix, never downstream D performance.

## Residual structure and covariance steering

`misc/diagnose_spline_residuals.py` tests what survives the first mean-curve
subtraction. A fixed, label-independent projection removing the curve's entire
seven-dimensional span still permits RBF speed prediction at D R² 0.866.
Residual magnitude alone gives R² 0.024, while unit-normalized residuals retain
R² 0.902. Pairwise products of 64 PCA coordinates recover speed even after
label-independent span removal (R² 0.746). This supports interactions between
features as a substantial source of the remaining readability.

The subtraction audit also demonstrates a confound: subtracting the curve at
assigned speed from independently shuffled features creates RBF speed R²
0.879–0.898 from chance-level starting features. This does not quantify how
much of the real residual signal is injected, but shows why label-dependent
residual readability alone is insufficient evidence of original redundancy.

`misc/covariance_velocity.py` fits regularized, speed-dependent residual second
moments on A and selects dimension, bandwidth and shrinkage by B Gaussian
likelihood. It transports source residuals to target covariance, combines that
change with the original centroid mean difference, and evaluates the actual
encoder/predictor. Shape-only, norm-matched random, frozen mean-only and frozen
iterated linear conditions are retained. No new clips are generated.

For +0.50 m/s, covariance transport increases response from 5.39% to 6.91%,
but uses larger edits and increases context-position drift from 0.298 to
0.546 px. Rescaling to each clip's mean-only edit norm gives 5.46%, with paired
gain difference 0.08 percentage points (95% CI −0.68 to 0.82), and 0.402 px
drift. The equal-norm −0.50 condition shows a small gain of 1.04 percentage
points (CI 0.09–1.89, uncorrected for multiple conditions). Iterated linear
steering remains stronger at approximately 18%. Conditional whitening lowers
RBF speed R² in the modeled coordinates from 0.808 to 0.186, but full-residual
R² remains 0.676; improved distribution modeling does not by itself solve
predictor steering.

See the [combined diagnostic and steering report](results/covariance_velocity_v1/REPORT.md)
for results, uncertainty and limitations. To reproduce in fresh output
directories, run `misc/diagnose_spline_residuals.py` and `misc/run_covariance_velocity.sh`
on the GPU allocation. The runner includes whitening and equal-edit-norm
audits and the report. Existing fit directories are protected against overwrite.

## Multi-probe steering

Section 4 edits pooled block-19 representations using the saved nullspace probe
sequences. Independent ridge evaluators are fitted on C, with regularization chosen
by cross-validation entirely within C. Fixed targets and probe counts are evaluated
on D. The notebook reports target errors, construction residuals, and edit sizes;
it does not propagate edits through the encoder or generate changed videos.

To run steering without repeating earlier fits, restart the kernel, run the setup
cell and the cached-feature loading cell with `RECOMPUTE_FEATURES = False`, then
run Section 4. It reads `results/nullspace/` and saves its protocol, evaluator
weights, per-clip predictions, scores, and plots under `results/steering/`.
Rerunning Section 4 overwrites those steering outputs. Cached steering uses CPUs;
no model loading or GPU computation is needed.

## Spline steering

Section 5 fits a 64-dimensional PCA subspace on A and interpolates binned concept
centroids with periodic (direction) or natural (scalar) cubic splines. B selects
among 8, 12, and 16 knots using label-conditioned reconstruction error. It then
shows the presentation's offset-preserving edit and checks its predictions against
the frozen follow-up artifacts. Saved C-trained evaluators measure transfer on D.

Run Section 1, cached-feature loading, and Section 5 without refitting previous
probes. Section 4's saved evaluators/protocol/scores and `results/spline_offset_v1/`
must be present. To replay only the offset calculation, skip Section 5's fitting
cells and start at “Apply the offset-preserving edit to the frozen D examples.”
`results/spline/` contains the fitting protocol, validation choices, PCA/spline
parameters, per-clip path predictions, geometric diagnostics, and comparison plots.
Rerunning the fitting cells overwrites the historical model/protocol artifacts.
Appendix C retains the original PCA-replacement evaluation and overwrites its
evaluation outputs when run, with unchanged calculations. All unordered pairs of
the fixed Section 4 targets are evaluated at 21 waypoints with shared endpoints.
Appendix B separately recomputes the presentation's offset-preserving path errors
from saved predictions. Both remain pooled-representation experiments.

## Offset-preserving pooled PCA follow-up

`misc/evaluate_pooled_spline_offset.py` reuses the original saved PCA/centroid curves,
C-trained evaluators, D clips and fixed targets, changing only the edit to
`h_new = h + [f(target) - f(known_source)] @ PCA_components`. Run on a compute
node with the existing interpreter and a fresh `--output` directory; no GPU
inference or fitting is needed. Historical spline outputs remain untouched.

The offset-preserving edit increases direction/speed/acceleration target MAE
from 2.636° / 0.1783 m/s / 0.3843 m/s² to 5.107° / 0.1907 m/s / 0.4247 m/s².
It preserves the PCA residual exactly, makes smaller edits, and is identity
when source equals target. The presentation uses only the offset-preserving
spline results; the historical comparison remains in the experiment report. See [the matched comparison report](results/spline_offset_v1/REPORT.md)
for paired uncertainty estimates, checks, path comparisons and limitations.
The existing predictor experiment already uses offset preservation and is unchanged.
