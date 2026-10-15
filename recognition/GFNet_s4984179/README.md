# GFNet and CNN for confidence-aware ADNI classification

COMP3710 · s4984179. This research compares a scratch GFNet with a simple
convolutional baseline on prepared 2D MRI slices. The development record below
explains the data, mathematical operations, engineering choices and their
observed consequences. It distinguishes intended experiments from measurements.

## V1 development: patient sampling and aggregated selection

Each network predicts a class for one slice. V1 also constructs scan/patient
summaries and uses weighted scan NLL for selection; the consequences of that
additional objective are investigated as evidence becomes available.

### Recover actual patients and audit the supplied image folders

The source contains 30,520 grayscale JPEG slices from 1,526 acquisitions and
680 patients. Each acquisition contributes twenty prepared slices. The number
before the filename's underscore identifies an acquisition, not a patient;
the `ADNI_..._S_...` token in source metadata supplies the actual patient ID.
Labels map NC to 0 and AD to 1 (the source's AD metadata value is 2).

The supplied train/test folders share 216 patients even though their scan IDs
do not overlap. Different visits from the same person therefore cannot be
assumed independent. The inventory checks diagnosis consistency, dimensions
and recoverable identities before deciding how to divide the data. Folder
names remain provenance and are never model inputs. These counts come from
the saved inventory audit; no clinical diagnosis is inferred from a filename.

### Separate patients into four reproducible data roles

Patients, rather than images or acquisitions, define the partition boundary.
Sorted AD and NC patient strata use one private PCG64(2026) stream. The fixed
AD allocation is 133/22/22/44 and NC is 275/46/46/92 across TRAIN, TUNE,
CALIBRATION and TEST. All visits and slices inherit their patient's role.

This gives 408/68/68/136 patients and 18,560/2,860/3,180/5,920 slices.
TRAIN updates weights, TUNE guides development, CALIBRATION fits probability
or rejection parameters, and TEST measures the declared final procedure.
Keeping all four roles in a manifest prevents a later model from quietly
using a different population. Correlated slices still do not become thousands
of independent patients merely because the network consumes individual images.

### Load native grayscale slices without resizing

The observed image canvas is width 256, height 240. The loader returns
float32 tensors `[1,240,256]` by dividing grayscale values by 255, preserving
native resolution, aspect ratio and the supplied brain preparation. Converting
to RGB would repeat information, while square resizing would alter geometry.
Neither is required for a scratch model.

Only the image enters the network. Labels, patient IDs and relative paths
remain separate metadata for accounting and analysis. The manifest fixes
labels and partition membership; path resolution stays under the caller's
data root. No pretrained normalization, cropping or augmentation is introduced
in this initial baseline, so their effects cannot be confused with architecture.

### Sample one acquisition per patient for the initial training design

The initial rationale is to prevent frequent visitors from dominating the
training stream. For every epoch, choose one acquisition per TRAIN patient,
retain its twenty slices, and shuffle the concatenated slices. Independent
PCG64 streams determine visit choice and slice order from seed and epoch.
Reconstructing the sampler therefore does not consume the model's RNG.

This choice gives 8,160 presentations and 128 batch-64 updates, including a
final partial batch. It gives patients equal opportunity rather than giving
all supplied slices equal opportunity. That distinction is a hypothesis with
consequences for class and visit weighting, not an automatic improvement over
ordinary image sampling. Evaluation loaders keep every declared slice.

### Define scan probability means and patient-weighted metrics

V1 extends image probabilities into an acquisition summary by averaging its
twenty AD probabilities. A patient with V acquisitions gives each scan weight
1/V, making each patient contribute total mass one. The weighted scan NLL is
then intended to measure probability reliability without letting repeated
visits dominate the metric.

This operation has a different statistical unit from slice accuracy. Averaging
can cancel a confident error against other slices, and applying NLL after an
average is not the same as averaging slice NLL. The code keeps that weighted
objective explicit rather than calling it image accuracy. This distinction
will later become central to deciding whether V1 answers the assessment task.

### Build local CNN stages with small convolution kernels

The CNN provides a simple spatial baseline. Each stage uses two padded 3×3
convolutions, GroupNorm and GELU, then 2×2 max pooling. Two small kernels add
nonlinear processing while expanding local context gradually; pooling reduces
later computation. GroupNorm avoids depending on large per-device batch
statistics and has the same inference behavior with a short final batch.

Four stages use 32, 64, 128 and 256 channels. Global spatial averaging and a
256-to-2 linear head give raw NC/AD logits, with no softmax inside the model.
The 1,173,154 parameters are much fewer than GFNet's later configuration, so
this is a practical baseline, not a parameter-matched isolation of one operator.

### Implement Fourier filtering on the rectangular patch grid

A 240×256 image divided into 16×16 patches has a 15×16 grid, not a square.
The GlobalFilter reshapes tokens to that explicit grid, applies orthonormal
rfft2 over its two spatial axes, multiplies learned complex coefficients,
then uses irfft2 with the original spatial size. The real FFT has a 15×9
frequency grid; real and imaginary parameters are stored separately.

The operation filters learned features, not MRI acquisition k-space. Taking
sqrt(token_count) from a square-image reference would give the wrong geometry.
FP32/complex64 and explicit axes make the mathematical adaptation inspectable.
The implementation follows Rao et al.'s global-filter mechanism, with task
adaptations for grayscale input and rectangular dimensions.

### Embed slices and add residual global-filter blocks

A stride-16 convolution projects each patch into 256 channels. Learned
positional embeddings distinguish patch locations. Twelve independent blocks
use the reference single residual path: normalization, global filter,
normalization and a channel MLP expanding 256→1024→256 with GELU.

Stochastic depth increases from zero to 0.1 across blocks, dropping whole
residual branches per example during training. Evaluation disables it. The
final LayerNorm, mean over the 240 within-image tokens and two-class head
produce one prediction per slice. This token mean is an internal feature
operation; it is distinct from averaging predictions from different slices.

### Complete the scratch GFNet classifier and tensor-flow explanation

The complete scratch model has 7,276,802 real trainable scalars. Linear and
positional parameters use truncated-normal initialization; LayerNorm starts
with unit scales and zero offsets. There is no class token, external encoder
or pretrained checkpoint. Native rectangular geometry reaches the classifier
without an image resize.

![Native CNN and GFNet tensor flow](figures/model_architecture.png)

The diagram shows implemented tensor sizes rather than an unmodified RGB
paper figure. CNN emphasizes local spatial processing; GFNet mixes patch
features globally. Differences in capacity and regularization remain part of
the complete-system comparison and prevent a single-factor causal claim.

### Configure AdamW and complete effective-batch updates

The optimizer uses AdamW with betas (0.9,0.999), weight decay 0.01 for matrices
and complex filters, and zero decay for biases, norm vectors and positional
embeddings. Cross-entropy consumes raw logits and integer NC/AD targets.
Global gradient norm clipping at 1.0 limits finite spikes; clipping frequency
is diagnostic and is not itself proof of divergence or a smaller AdamW LR.

The effective batch is 64. A short final batch uses its actual size, and
sample-weighted epoch loss avoids giving the final small batch the same
influence as a full batch. Python, NumPy and Torch seeds are explicit so model
initialization and sampling can be compared under a stated configuration.

### Save compact epoch boundaries and reproducible random state

A complete boundary needs more than model weights: optimizer moments, global
update cursor, epoch, selected best state, patience and Python/NumPy/Torch/CUDA
RNG states determine continuation. A separate selected checkpoint supports
inference without applying optimizer or training RNG state. Temporary-file
replacement avoids publishing a half-written file as a completed boundary.

The compact representation keeps scientific continuation state and readable
configuration/history. Repeated validation scripts and operational submission
receipts are not part of the learning algorithm. Historical training results
remain associated with their saved execution sources; simplifying the code
here does not rerun those experiments or prove byte-equivalent checkpoints.

### Establish allocated A100 execution and initial resource feasibility

Native FP32/complex64 execution uses an actual allocated A100 and deterministic
operations with TF32 disabled. Hardware is identified from the visible device,
not inferred from a partition label. Runtime records include software, device
identity and thread settings; reproducibility remains scoped to that setup.

Saved job 624327 passed the initial GPU phases. Batch 64's allocated training
peaks were 8.697 GiB for CNN and 2.603 GiB for GFNet, against 39.081 GiB
initially free. The preliminary batch-of-twenty inference benchmark measured
median/p95 8.257/8.450 ms for CNN and 8.005/8.181 ms for GFNet. It included
forward, softmax and probability mean, excluding image decode and transfer;
it was not single-slice end-to-end latency or trained-model accuracy.

### Record five-epoch pilots and exact boundary continuation

Saved job 625831 completed five control epochs and a fresh-process continuation
for each model. The saved model/optimizer/RNG/progress comparisons matched
exactly. New ADNI work was 1,536 updates, 97,920 TRAIN and 34,320 TUNE
presentations, counting only newly executed work rather than inherited histories.

The measured steady epoch medians were 43.74 seconds for CNN and 42.37 seconds
for GFNet under the sampled training recipe. Typical 60-epoch extrapolations
were 43.93 and 42.59 minutes, with larger cautious estimates; they were not
completed formal-fit durations. These results support native batch-64 execution
and boundary resume, while classification capability and confidence reliability
remain separate research questions.

### Explain feasibility, research priorities and measured limits

The feasibility review joins four considerations: patient-disjoint data,
rectangular architecture implementation, measured A100 resources, and testable
classification/reliability questions. The evidence supports an executable
prototype; it does not support a clinical claim or an 80% accuracy guarantee.

Confidence must be examined alongside correctness, including confident wrong
predictions and false negatives. Classification, calibration, selective
prediction, real error examples and computational cost should jointly inform
an engineering recommendation. The one-page review and later tutor clarification
are evidence documents, not a reason to call unfinished experiments complete.

### Define a global cosine schedule and explicit NLL stopping policy

The formal horizon is 60 sampled epochs, 7,680 updates. Warmup uses 640
updates, followed by cosine decay to 1% of peak. The global update index,
rather than the process's local epoch count, prevents resume from restarting
the schedule. Candidate peaks are 1e-4, 3e-4 and 1e-3.

Strict-best weighted scan NLL selects checkpoints. A separate patience anchor
requires an improvement of at least 1e-4; small strict-best improvements do
not continually reset patience. Stop after at least thirty epochs with twelve
non-improving epochs, or at sixty. This makes selection and stopping reproducible,
but it does not establish that weighted scan NLL is the right assessment target.

### Integrate training, full TUNE evaluation and continuation

The integrated fit reconstructs the patient's visit sampler for each epoch,
updates on its 8,160 slices, and evaluates every one of the 2,860 TUNE slices
with fixed weights and no gradients. TUNE records then produce both raw-slice
diagnostics and the declared aggregated selection statistic.

Resume restores the last completed boundary, optimizer, RNG, schedule cursor
and controller rather than creating a replacement seed or fit. An invocation
may run only a segment of the global horizon. This distinction matters when
Slurm wall-time or storage limits require segmentation; a saved history must
not be counted as new training when it is loaded again.

### Record native A100 continuation of the formal training core

Saved job 629388 passed six phases and four learning invocations on A100.
It executed 768 new updates with 48,960 TRAIN and 17,160 TUNE presentations.
Saved CPU comparisons found exact functional continuation; the recovery did
not repeat learning to manufacture a successful comparison.

CNN/GFNet maximum allocated memory remained about 8.697/2.603 GiB. Recorded
complete epoch times included checkpoint persistence and differed from the
earlier pilot timings. This is why the cost discussion separates measured
scopes and avoids treating one preliminary median as a guaranteed full-fit
or inference latency.

### Reserve development fits and predeclare the LR selection barrier

The experiment plan reserves six development fits: two architectures × three
peaks with seed 3710. Only after all six finish does V1 select each model's
minimum best weighted scan NLL, breaking exact ties by the declared LR order.
Four conditional fits add seeds 3711 and 3712 at those selected peaks.

The winning seed-3710 fit is reused, limiting the total to ten fits rather
than retraining the favorable run. Patient populations and schedules must be
consistent between candidates, including their complete declared horizons.
These controls prevent accidental candidate replacement, but the scientific
validity of the selection objective still needs examination.

### Expose one direct fit command with explicit pause and resume

The executable training entry chooses model, LR, seed and output directory,
reads the fixed manifest and initializes its allocated runtime before model
construction. Fresh output directories prevent silently mixing histories.
The same command with resume continues the saved experiment and global horizon.

The compact interface expresses one fit rather than embedding a scheduler
framework inside the trainer. Cluster submission scripts, queues and recovery
records stay external. A command that exits after a requested segment can
leave an active paused fit; that is not a terminal training result and must
not be treated as a completed LR candidate.

### Fit scan temperature and a fixed rejection policy on CALIBRATION

V1 fits one positive temperature per model/seed on CALIBRATION scan means,
using the fixed 201-point geometric grid from 0.05 to 20 plus 1. The chosen
value minimizes weighted NLL; exact ties prefer closeness to one, then smaller
temperature. This changes probability scale, not the learned feature extractor.

The highest confidence threshold reaching at least 80% patient-weighted scan
coverage is then frozen, including all exact-confidence ties. Exact rational
1/V masses avoid changing eligibility through floating-point summation.
Evaluation applies the same temperature/threshold without refitting. Coverage
is a retained-prediction proportion, not an accuracy requirement, and scan
calibration does not establish calibrated slice probabilities.

## Artificial Intelligence Usage Disclosure

OpenAI Codex assisted with implementation, code organization, interpretation
of saved experiment records and documentation. Model choices and results remain
the author's responsibility. Saved measurements are identified by their run or
job; writing or simplifying code does not reproduce those experiments.

## References

- Rao et al., *Global Filter Networks for Image Classification*, NeurIPS 2021.
  [Paper](https://arxiv.org/abs/2107.00645), [reference code](https://github.com/shakes76/GFNet).
- Wu and He, *Group Normalization*, ECCV 2018.
  [Paper](https://arxiv.org/abs/1803.08494).
- Loshchilov and Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019.
  [Paper](https://arxiv.org/abs/1711.05101).
- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017.
  [Paper](https://proceedings.mlr.press/v70/guo17a.html).
