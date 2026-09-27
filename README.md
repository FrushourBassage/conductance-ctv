# conductance-ctv

Hydraulic morphometry and four-type classification of iliac vein lesions on
venous-phase computed tomography venography.

The release implements the measurement task the manuscript defines rather than a
label-prediction task. For every millimetre along the iliofemoral centreline it
slices the lumen mask orthogonally, solves the unit-source Dirichlet problem on
that cross-section, and reports the conductance `G = integral of u dA` together
with the shape factor `kappa = A^2 / (8 pi G)`, the counted channel number and the
series resistance of the span. Those per-segment descriptors are fused with a
20 mm tube of backbone features to give a calibrated four-type call and a
interpretable severity ratio, and the whole measurement is evaluated by segment,
by limb and by patient.

## What each part of the tree does

| Path | Purpose |
| --- | --- |
| `src/conductance_ctv/hydraulics/` | centreline, orthogonal slicing, the Dirichlet solve, the Pólya–Szegő bounds, the descriptor vector |
| `src/conductance_ctv/models/` | encoder families, the anisotropic decoder, the descriptor and appearance branches, the fusion head, the comparator models |
| `src/conductance_ctv/data/` | NIfTI and mask I/O, the grid schema, patient-level splits and cross-fitting, the schema-compatible cohort fixture |
| `src/conductance_ctv/losses/` | confluence-weighted Dice and cross-entropy, class-balanced multinomial loss, severity regression |
| `src/conductance_ctv/metrics/` | Dice, HD95, centreline Dice, macro-AUROC and macro-F1, Cohen's kappa, ICC(2,1), Bland–Altman, DeLong, McNemar, NRI, decision-curve net benefit |
| `src/conductance_ctv/training/` | optimisers, schedules, mixed precision, EMA, atomic checkpoints, the two trainers, patient-level cross-fitting |
| `src/conductance_ctv/evaluation/` | the evaluation protocol with patient-clustered resampling, the association models, the seven reported tables |
| `src/conductance_ctv/verification/` | the claim register, the executed checks, the three release artefacts |
| `src/conductance_ctv/cli/` | `train`, `evaluate`, `hydraulics`, `architectures`, `verify` |
| `configs/` | one experiment file per reported experiment and ablation, plus the model, data and training defaults they inherit |

## Environments

Python 3.12 or newer.

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

```
conda env create -f environment.yml
conda activate conductance-ctv
```

```
docker build -t conductance-ctv .
docker run --rm -v "$PWD/reports:/opt/conductance-ctv/reports" conductance-ctv \
  python -m conductance_ctv.cli.evaluate experiment=main
```

The image pins the CUDA runtime the reported GPU-hours were run against; the CPU
path is the same image with `--device cpu`.

## Imaging data

Two public corpora are used, both for pretraining only. Neither supplies an
evaluation volume and neither carries a clinical variable.

| Corpus | Version | Licence | Role |
| --- | --- | --- | --- |
| TotalSegmentator computed tomography dataset | 2.0.1, pinned | CC BY 4.0 | abdominal vascular labels (`inferior_vena_cava`, `iliac_vena_left`, `iliac_vena_right`) |
| AbdomenAtlas computed tomography dataset | 1.0 | the licence stated at the repository | abdominal volumes and organ labels |

`dataset_urls.txt` records the resolvable source of each, with the role and what
was checked when the list was written. Fetch each corpus into a directory of your
choice and point `configs/data/*.yaml` at it; nothing in the release bundles or
redistributes them. Re-verifying the list is a separate, explicit step:

```
python -m conductance_ctv.cli.verify --probe-live
```

Without that flag the verification pass reads the shipped list and opens no
socket, so its verdict follows from the repository rather than from what the
machine running it could reach.

The clinical cohort of the study is not shareable: it consists of de-identified
imaging records that cannot be released without re-identification risk. The
release therefore ships a cohort fixture with the same schema and a ground truth
prescribed in closed form, which is what every config in `configs/` resolves by
default. All cohort-level values of the manuscript are marked `NOT_RUN` in the
verification report for that reason, and no number in this tree should be read as
one of them.

## Hydraulic signature of a limb

Solve the section problem on a described geometry:

```
python -m conductance_ctv.cli.hydraulics --shape ellipse --a 20 --b 6 --step 0.5
python -m conductance_ctv.cli.hydraulics --shape septum --channels 3 --area 1800
```

Or on a real mask, reported per segment with the minimum luminal area, the
arc length of the minimum-conductance sample, the shape factor, the channel count,
the series resistance, the dimensionless ratio and the conductance-weighted
length:

```
python -m conductance_ctv.cli.hydraulics --mask limb_mask.npz --spacing 1.0 0.6 0.6
```

The channel count and the shape factor are meant to be read together: a shape
factor above the counted channel number means sub-resolution septa, not a
simple section.

## Parameter budgets

Table 1 of the manuscript reports each family's parameter total without its
per-layer widths. The widths are this release's free variable, and
`resolve_width_bracket` returns the nearer of the two achievable totals around the
reported one; the measured count is always reported beside the reported one.

| Encoder | Reported | Width multiplier | Measured |
| --- | --- | --- | --- |
| `resnet50_unet` | 46.3 M | 0.9531 | 46.08 M |
| `densenet121_unet` | 28.7 M | 1.5625 | 29.74 M |
| `attention3d_unet` | 38.1 M | 1.2266 | 38.35 M |
| `foundation_trunk` | 131.0 M | 3.8477 | 131.02 M |
| `universal_transferred` | 96.4 M | 2.1328 | 96.22 M |
| `universal_prompted` | 142.0 M | 2.5859 | 141.55 M |

DenseNet is the widest ladder of the six: no multiplier lands within one percent
of 28.7 M, because the growth rate moves in integer steps, and 29.74 M is the
nearer of the two achievable neighbours. The proposed configuration reports
133.6 M for the whole network, and its declared decoder width multiplier
(`configs/model/foundation_trunk.yaml`) is resolved so the built network reaches
133.62 M.

```
python -m conductance_ctv.cli.architectures --resolve
```

## Training

One command per configuration; the segmentation backbone and the type model are
trained separately and the temperature is fitted on the tuning partition alone.

```
python -m conductance_ctv.cli.train experiment=main stage=segmentation
python -m conductance_ctv.cli.train experiment=main stage=type
```

`configs/train/main.yaml` carries the reported profile: 200 epochs, batch 8 with
a gradient accumulation of 2, AdamW at 3e-4, a warmup-cosine schedule, bf16, a
gradient clip of 1.0 and an EMA decay of 0.999. The manuscript states the
parameter total and the 214 A100 GPU-hours but not the batch size, learning rate,
schedule, epoch count or precision; every value it does not state is an
engineering default and is listed as such in `claim_to_code.json`.

## Evaluation and the reported tables

```
python -m conductance_ctv.cli.evaluate experiment=main --out reports
```

Each run writes its per-table JSON and a plain-text rendering into `reports/`.
The experiment files map onto the manuscript as follows.

| Paper artefact | Configuration |
| --- | --- |
| Table 1 segmentation columns | `comparator_endtoend_*`, `comparator_frozen_linear`, `comparator_universal_*`, `main` |
| Table 2 component necessity | `ablation_no_hydraulic_operator`, `ablation_no_channel_count`, `ablation_no_normalisation`, `ablation_no_star_pooling`, `ablation_no_appearance`, `ablation_no_collateral`, `ablation_no_adaptation`, `ablation_no_confluence_loss` |
| Table 2 data efficiency | `ablation_init_10pct_scratch`, `ablation_init_10pct_pretrained`, `ablation_init_25pct_pretrained`, `ablation_init_100pct_scratch` |
| Table 3 comparators | `comparator_single_diameter`, `comparator_radiomics_*`, `comparator_endtoend_*`, `comparator_universal_*`, `fairness_debiased`, `main` |
| Table 4 transport | `supplementary_transport`, `supplementary_phase_subgroups`, `supplementary_vendor_subgroups`, `supplementary_leave_one_site_out` |
| Table 5 morphometry and cost | `main` |
| Table 6 clinical stages | `supplementary_reader_arm` |
| Table 7 association | `main` |

Values the manuscript reports and this release is built to reach, quoted here as
targets rather than as results: macro-AUROC 0.941 on the development sites and
0.917 externally against a pre-specified floor of 0.900; ICC(2,1) 0.941 for the
resistance ratio and 0.937 for the minimum luminal area at the common iliac vein;
kappa 0.842 against the two-reader consensus; a reader-accuracy change of +0.041
under assistance against a non-inferiority margin of 0.05 in kappa. The external
floor, the ICC floor, the management-relevant change floor of 0.10 and the
non-inferiority margin are recorded together in
`src/conductance_ctv/verification/checks.py` and are applied to the tuning
partition before the sealed partition is scored.

## Compute

| Quantity | Value |
| --- | --- |
| GPU | NVIDIA A100, 40 GB |
| Training | 214 A100 GPU-hours for the reported run |
| Total parameters | 133.6 M |
| Inference, full pipeline | 8.7 s per volume |
| Hydraulic solve | 41 ms per segment |
| Reader time | 92 s unassisted, 71 s assisted |

The GPU-hours are the manuscript's figure for the reported run. Executing that run
requires an accelerator; without one, the release executes its complete pipeline
on the cohort fixture at reduced width, which is what the verification pass does.

## Verification

Two passes are run and both write into this directory.

```
python -m conductance_ctv.cli.verify
```

The first pass is the manuscript-claim to code mapping in `claim_to_code.json`: one
record per claim, each naming the paper location it comes from, the files and
symbols that carry it, the check that decides it, and the status that check
returned. The second pass is the execution pass recorded in
`verification_report.json`: data reading, the forward pass, the loss, the backward
pass, a parameter update, a checkpoint round trip, a single-batch overfit and a
two-step training run, each with the evidence that produced its verdict.

Statuses are `PASS`, `FAIL`, `NOT_RUN` or `BLOCKED` and nothing else. A check that
cannot be attempted in the current environment is `NOT_RUN` or `BLOCKED`, never
`PASS`: the cohort-level values of the manuscript are `NOT_RUN` because the
clinical cohort is not shareable, the GPU-scale run is `NOT_RUN` without an
accelerator, pretraining is `NOT_RUN` because the public corpora are not staged,
and the container build is `NOT_RUN` without a container runtime.
`integrity_manifest.json` digests every shipped file and excludes only itself, so a
later edit to any file shows up as a digest mismatch on the next run.

## Licence

MIT, see `LICENSE`. Third-party datasets and packages keep their own terms, see
`NOTICE`.
