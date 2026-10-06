# Running the CorrNet experiments

Run these commands from `/model` inside the container after completing the environment and dataset setup. The container script mounts the `CorrNet/` folder itself at `/model`, so `activelearning.py` is already sitting there and you don't need to change directories.

The examples in this documentation use CorrNet, but the exact same commands work for Swin-MSTP too! The Swin-MSTP container mounts `Swin-MSTP/` at its own `/model` in the same way, and runs under its own container name.

```bash
python3 activelearning.py --help
```

The script you should run is called `activelearning.py`. It automatically creates and manages the labeled and unlabeled pools, trains the model, extracts the features needed for sample acquisition, moves the selected videos into the labeled pool, and prepares the next iteration.

## The recommended workflow is to run Random Sampling first, then everything else

Before getting into the individual arguments, it's worth explaining how a full comparison is meant to be run.

To make the comparisons fair, every acquisition strategy has to start from the **same initial labeled pool** and see the **same features** on its first query. Otherwise you can't tell whether a method won because it selects better samples or just because it got a luckier random initialization. The paper handles this by using identical random seeds across methods, and by having every method branch off a shared starting point.

In this code, that shared starting point is produced by **a Random Sampling run**. There's a reason it has to be Random: the `random` branch of `activelearning.py` only extracts features on its very first iteration (after that it just samples video IDs and doesn't need features at all). So the first Random iteration gives you, in one shot, everything the other methods need to start from:

- a trained model (`_best_model.pt`),
- the labeled and unlabeled pools of data,
- extracted features for **both** pools.

Run that first Random Sampling with `-b 5` so its features include the Monte Carlo dropout passes. Every strategy can then branch off it, UNCAST included. See [`-b`](#arguments) below for what this costs in disk space, and for when `-b 0` is the better choice.

So the intended order is:

```text
1. Run Random Sampling on the full schedule.
       │
       │  Its iteration 1 produces the shared initial pool + model + features.
       │
       ├──> 2. Run RAIDAL           from that checkpoint  ─┐
       ├──> 2. Run Entropy          from that checkpoint   │  All start from the
       ├──> 2. Run UNCAST           from that checkpoint   │  same starting point, then
       ├──> 2. Run Core-set         from that checkpoint   │  diverge after their
       └──> 2. Run CTC-BADGE, ...   from that checkpoint  ─┘  first query.
```

Only the **first** acquisition of each method reuses the shared features. After that first selection, each method's labeled pool has diverged, so a new model is trained and a fresh set of features is generated from that method's own pool. Therefore the starting line is fixed, but each method is free to branch off from that and build its own pool of data.

The Random Sampling run also doubles as the Random baseline itself, so nothing is wasted.

## Arguments

Hint: You can run `python3 activelearning.py --help` to get a quick help summary.

`-d` points to the original prepared dataset. Should be either `dataset/phoenix2014`, `dataset/phoenix2014-T` or `dataset/Isharah`.

`-f` controls the frame budget. On a fresh run, the first value creates the initial random labeled pool and the remaining values are acquisition increments. When starting from `--checkpoint`, every value in `-f` is an acquisition increment because the initial labeled pool already exists.

So for example: if you run `python3 activelearning.py -f 10000 15000 10000 -s random [other arguments]`, the first cycle will randomly sample videos until the total number of frames in the labeled pool totals 10k frames. Then the second cycle will contain 25k frames, since we're adding 15k frames on top of the initial 10k. And then the third cycle will contain 35k frames, since we're adding 10k extra frames on top of the 25k that were in the second cycle.

If you use `--checkpoint`, the script automatically detects the number of frames that were in that checkpoint's labeled pool. So for example, if you run `python3 activelearning.py -f 10000 15000 10000 -s raidal --checkpoint exampleCheckpoint [other arguments]` and the checkpoint's labeled pool contains 50k frames, you'll be adding 10k extra frames on top of the 50k frames in the checkpoint, totalling 60k frames. Then 75k and finally 85k frames.

There's no separate flag for the number of acquisition cycles, since the script automatically derives it from how many values you passed to `-f`.

`-w` selects the experiment output directory.

`-n` gives the generated labeled and unlabeled dataset copies a unique experiment name.

`-s` selects the acquisition strategy.

`--device` selects the GPU index.

`--config` selects the CorrNet training configuration. Notice I included the training configurations used for my experiments in the `configs/` folder. `baseline-replicationX.yaml` should be used with PHOENIX-2014 and `baseline-replicationX-phoenix-T.yaml` should be used with PHOENIX-T. The `baseline-replicationX-Isharah.yaml` configs ship in `Swin-MSTP/configs/` instead, since Swin-MSTP is the backbone the paper uses for Isharah.

`-b` controls how many Monte Carlo dropout inference passes are saved during feature extraction. With `-b 0` each video is passed through the model once, with dropout disabled. With `-b 5` the model additionally runs five stochastic passes with dropout enabled, and every one of those passes is saved alongside the clean one.

Those extra passes are what UNCAST uses to measure model uncertainty, and they are the only reason to raise `-b` above zero. They are not free, since each pass stores another full copy of the per-frame classifier logits for every video, which can quickly add up to a lot of used disk space.

The recommendation is to use `-b 5` on the Random Sampling run that generates the shared starting point, because that makes its feature directories usable by every strategy, UNCAST included. If you already know you aren't going to run UNCAST, or any other strategy that needs dropout outputs, you can use `-b 0` instead. Nothing with dropout is saved, and the extracted features will use less disk space.

Note that `-b` applies to every acquisition cycle, not just the first one. Once a method branches off and starts training its own models, its later feature extractions use the `-b` value that method was launched with.

Lastly, when using `-b` above zero, the parser requires at least two passes, so `-b 1` is rejected.

`--checkpoint` resumes from a previous labeled iteration work directory.

`--use-model` reuses the checkpoint's trained model but regenerates the features.

`--use-features` reuses both the checkpoint's model and its pregenerated features. This is the flag that makes the shared-start workflow work.

### How `-f` maps to reported budgets

The number of loop iterations isn't quite the number of points you get in a results table, so it's worth spelling out.

On a **fresh run**, the first value in `-f` builds the initial random pool and iteration 1 trains on it. Each later iteration trains on the pool after one more acquisition. So the 10-value PHOENIX-2014 schedule below gives you the initial 20k baseline plus the 9 reported budgets.

On a **checkpoint run**, the first iteration doesn't train (it already has the checkpoint's model), it just performs the first acquisition. The script accounts for this by running one extra loop iteration internally, so the 9-value schedule still lands on the same 9 reported budgets. The leading initial-pool value is simply dropped, since that pool already exists.

## Strategy names

These are the strategy names of the main methods.

```text
random
raidal
global-pooled-coreset
soft-attention-coreset
entropy
uncast
ctc-badge
```

These are the names of the methods in the ablations:

```text
unfiltered-raidal
raidal-E
raidal-K1
raidal-K2
raidal-K5
raidal-R0
raidal-R2
raidal-R3
raidal-R4
```

And the additional CTC-BADGE experiment from the supplementary material:

```text
raidal-ctc-badge
```

| Strategy name | Description |
|---|---|
| `random` | Random Sampling baseline. Also the feature generator for the shared-start workflow. |
| `raidal` | RAIDAL, canonical setting (top 10 hypotheses, R = 1). |
| `entropy` | Mean per-frame entropy baseline. |
| `uncast` | UNCAST sequence-level CTC uncertainty. The only strategy that needs dropout features, so it needs `-b` set to at least 2 (the examples use `-b 5`, matching the paper configuration). |
| `global-pooled-coreset` | Core-set over one globally pooled embedding per video. |
| `soft-attention-coreset` | Timestep-level Core-set, weighted by non-blank probability. |
| `ctc-badge` | CTC-BADGE with gradient embeddings and k-means++ selection. |
| `raidal-ctc-badge` | Appendix E: CTC-BADGE with RAIDAL's TROI filter applied to the gradient embedding. |
| `unfiltered-raidal` | Ablation (a): RAIDAL with decoder-based filtering removed. |
| `raidal-E` | Ablation (b): RAIDAL's filter applied to Entropy. |
| `raidal-K1`, `raidal-K2`, `raidal-K5` | Ablation (c): number of decoded hypotheses used to build the TROI. |
| `raidal-R0` … `raidal-R4` | Ablation (d): Temporal Expansion Radius sensitivity. |

# Hands-On Examples

## PHOENIX-2014

The paper starts PHOENIX-2014 from 20k labeled frames and then evaluates 30k, 40k, 50k, 60k, 80k, 100k, 120k, 160k, and 200k frames.

### Step 1: Run Random Sampling (also generates the shared starting point)

Pass the initial budget followed by the acquisition increments.

```bash
python activelearning.py \
  -d dataset/phoenix2014 \
  -f 20000 10000 10000 10000 10000 20000 20000 20000 40000 40000 \
  -w work_dir/Phoenix2014/Random/Rep1 \
  -n RandomRep1 \
  -s random \
  --device 0 \
  -b 5 \
  --config configs/baseline-replication1.yaml
```

`-b 5` saves five Monte Carlo dropout passes per video on top of the clean one. That makes the resulting features usable by every strategy, including UNCAST, at the cost of larger feature directories and a slower extraction. If you aren't going to run UNCAST, use `-b 0` here instead and the dropout outputs are never written.

When its first iteration finishes, these three directories are created and are what every other method will point to:

```text
work_dir/Phoenix2014/Random/Rep1/phoenix2014-RandomRep1-labeled-iteration1
work_dir/Phoenix2014/Random/Rep1/phoenix2014-RandomRep1-labeled-iteration1-features
work_dir/Phoenix2014/Random/Rep1/phoenix2014-RandomRep1-unlabeled-iteration1-features
```

If you have multiple GPUs available, you don't have to wait for the whole Random run to finish before starting the others. As soon as iteration 1 has produced those directories, the shared starting point is ready.

### Step 2: Run RAIDAL from the shared starting point

Notice that Random's checkpoint already contributes the initial 20k frames from Random's first acquisition cycle. Therefore `-f` should contain only the nine remaining acquisition budgets. This way RAIDAL's frame budgets line up to Random's 30k, 40k, 50k, etc.

```bash
python activelearning.py \
  -d dataset/phoenix2014 \
  -f 10000 10000 10000 10000 20000 20000 20000 40000 40000 \
  -w work_dir/Phoenix2014/RAIDAL/Rep1 \
  -n RaidalRep1 \
  -s raidal \
  --device 0 \
  -b 0 \
  --config configs/baseline-replication1.yaml \
  --checkpoint work_dir/Phoenix2014/Random/Rep1/phoenix2014-RandomRep1-labeled-iteration1 \
  --use-features \
    work_dir/Phoenix2014/Random/Rep1/phoenix2014-RandomRep1-labeled-iteration1-features \
    work_dir/Phoenix2014/Random/Rep1/phoenix2014-RandomRep1-unlabeled-iteration1-features
```

`--use-features` expects the two feature run directories, not their internal `train` and `test` folders. The code appends those split names itself. Always pass the labeled features first and the unlabeled features second.

Note that this run uses `-b 0` even though Step 1 used `-b 5`. This is because RAIDAL does not use dropout, so there is no reason to save video features with dropout when using RAIDAL. The same applies to every strategy in Step 3, with UNCAST being the only one that needs multiple dropout passes.

### Step 3: Run the remaining methods

Every other strategy is the same command with a different `-s`, a different `-w`, and a different `-n`. Only those three change:

```bash
python activelearning.py \
  -d dataset/phoenix2014 \
  -f 10000 10000 10000 10000 20000 20000 20000 40000 40000 \
  -w work_dir/Phoenix2014/Entropy/Rep1 \
  -n EntropyRep1 \
  -s entropy \
  --device 0 \
  -b 0 \
  --config configs/baseline-replication1.yaml \
  --checkpoint work_dir/Phoenix2014/Random/Rep1/phoenix2014-RandomRep1-labeled-iteration1 \
  --use-features \
    work_dir/Phoenix2014/Random/Rep1/phoenix2014-RandomRep1-labeled-iteration1-features \
    work_dir/Phoenix2014/Random/Rep1/phoenix2014-RandomRep1-unlabeled-iteration1-features
```

Swap `-s entropy` / `-w .../Entropy/Rep1` / `-n EntropyRep1` for any of:

```text
-s global-pooled-coreset    -w work_dir/Phoenix2014/Coreset/Rep1        -n CoresetRep1
-s soft-attention-coreset   -w work_dir/Phoenix2014/SoftAttn/Rep1       -n SoftAttnRep1
-s ctc-badge                -w work_dir/Phoenix2014/CTCBadge/Rep1       -n CTCBadgeRep1
-s unfiltered-raidal        -w work_dir/Phoenix2014/NoFilter/Rep1       -n NoFilterRep1
-s raidal-E                 -w work_dir/Phoenix2014/RaidalE/Rep1        -n RaidalERep1
-s raidal-R0                -w work_dir/Phoenix2014/RaidalR0/Rep1       -n RaidalR0Rep1
```

The experiment name in `-n` has to differ from the one in the checkpoint path, and has to be unique across runs. The parser rejects repeats on purpose to prevent accidentaly overwriting files.

### Replications

Simply run the whole Step 1 -> Step 2 -> Step 3 sequence three times, once per replication, changing the config and the paths:

```text
Rep1 → --config configs/baseline-replication1.yaml
Rep2 → --config configs/baseline-replication2.yaml
Rep3 → --config configs/baseline-replication3.yaml
```

Each replication needs **its own** Random Sampling run to branch from, because each config carries a different random seed. Don't branch a Rep2 method off the Rep1 checkpoint, otherwise the comparisons would no longer be paired.

Notice that each dataset has a unique configuration file. `baseline-replicationX.yaml` should only be used with PHOENIX-2014, while `baseline-replicationX-phoenix-T.yaml` should only be used with PHOENIX-T, and `baseline-replicationX-Isharah.yaml` should only be used with Isharah. You can check which dataset each file corresponds to by opening it and looking for the `dataset:` key.

## PHOENIX-T

The PHOENIX-2014-T experiments start from 20,684 frames. The query sizes used to reach the later budgets are 10,342 for the first four acquisitions, 20,684 for the next three, and 41,367 for the last two.

```bash
python activelearning.py \
  -d dataset/phoenix2014-T \
  -f 20684 10342 10342 10342 10342 20684 20684 20684 41367 41367 \
  -w work_dir/Phoenix2014T/Random/Rep1 \
  -n RandomRep1 \
  -s random \
  --device 0 \
  -b 5 \
  --config configs/baseline-replication1-phoenix-T.yaml
```

Then branch the other methods off its first iteration exactly as in Step 2 above, dropping the leading `20684` from `-f`:

```bash
python activelearning.py \
  -d dataset/phoenix2014-T \
  -f 10342 10342 10342 10342 20684 20684 20684 41367 41367 \
  -w work_dir/Phoenix2014T/RAIDAL/Rep1 \
  -n RaidalRep1 \
  -s raidal \
  --device 0 \
  -b 0 \
  --config configs/baseline-replication1-phoenix-T.yaml \
  --checkpoint work_dir/Phoenix2014T/Random/Rep1/phoenix2014-T-RandomRep1-labeled-iteration1 \
  --use-features \
    work_dir/Phoenix2014T/Random/Rep1/phoenix2014-T-RandomRep1-labeled-iteration1-features \
    work_dir/Phoenix2014T/Random/Rep1/phoenix2014-T-RandomRep1-unlabeled-iteration1-features
```

## Isharah

> The paper runs Isharah on Swin-MSTP rather than CorrNet, following the Isharah authors' observation that CorrNet struggles in low-data regimes on this dataset. Accordingly, the Isharah configs ship in `Swin-MSTP/configs/`, and `CorrNet/configs/` carries the two PHOENIX ones only. Run these commands from the Swin-MSTP container, or write your own Isharah config with `dataset: Isharah` if you want to try it on CorrNet.

Isharah starts at 200,000 frames, and progresses in increments of 100,000 frames until the labeled pool reaches 700,000. Run the first Random sampling iteration on Isharah with:

```bash
python activelearning.py \
  -d dataset/Isharah \
  -f 200000 100000 100000 100000 100000 100000 \
  -w work_dir/Isharah/Random/Rep1 \
  -n RandomRep1 \
  -s random \
  --device 0 \
  -b 5 \
  --config configs/baseline-replication1-Isharah.yaml
```

And the branched methods, dropping the leading `200000`:

```bash
python activelearning.py \
  -d dataset/Isharah \
  -f 100000 100000 100000 100000 100000 \
  -w work_dir/Isharah/RAIDAL/Rep1 \
  -n RaidalRep1 \
  -s raidal \
  --device 0 \
  -b 0 \
  --config configs/baseline-replication1-Isharah.yaml \
  --checkpoint work_dir/Isharah/Random/Rep1/Isharah-RandomRep1-labeled-iteration1 \
  --use-features \
    work_dir/Isharah/Random/Rep1/Isharah-RandomRep1-labeled-iteration1-features \
    work_dir/Isharah/Random/Rep1/Isharah-RandomRep1-unlabeled-iteration1-features
```

Apart from the config file, the commands are identical on either backbone.

## UNCAST

UNCAST is the only strategy that reads the Monte Carlo dropout outputs, so it's the reason Step 1 is run with `-b 5`. When using UNCAST, `-b` should be set to a non-zero value.

```bash
python activelearning.py \
  -d dataset/phoenix2014 \
  -f 10000 10000 10000 10000 20000 20000 20000 40000 40000 \
  -w work_dir/Phoenix2014/UNCAST/Rep1 \
  -n UncastRep1 \
  -s uncast \
  --device 0 \
  -b 5 \
  --config configs/baseline-replication1.yaml \
  --checkpoint work_dir/Phoenix2014/Random/Rep1/phoenix2014-RandomRep1-labeled-iteration1 \
  --use-features \
    work_dir/Phoenix2014/Random/Rep1/phoenix2014-RandomRep1-labeled-iteration1-features \
    work_dir/Phoenix2014/Random/Rep1/phoenix2014-RandomRep1-unlabeled-iteration1-features
```

### If the shared features were generated with `-b 0` and now you realize you need to run UNCAST

A feature directory written with `-b 0` contains no dropout outputs, so passing it to UNCAST through `--use-features` fails, since `uncast.py` looks for `classifierFeaturesWithDropout` and finds nothing.

You don't have to rerun Random Sampling for that. Swap `--use-features` for `--use-model`, which reuses the checkpoint's trained model and regenerates the features from it, this time with dropout enabled:

```bash
python activelearning.py \
  -d dataset/phoenix2014 \
  -f 10000 10000 10000 10000 20000 20000 20000 40000 40000 \
  -w work_dir/Phoenix2014/UNCAST/Rep1 \
  -n UncastRep1 \
  -s uncast \
  --device 0 \
  -b 5 \
  --config configs/baseline-replication1.yaml \
  --checkpoint work_dir/Phoenix2014/Random/Rep1/phoenix2014-RandomRep1-labeled-iteration1 \
  --use-model
```

The initial labeled pool and the model are still the shared ones, so UNCAST still starts from the same place as everything else. Only the feature extraction is redone. It costs one extra extraction pass over both pools, which is still far cheaper than retraining.

## RAIDAL settings

The CorrNet PHOENIX experiments use the top 10 decoded hypotheses and a Temporal Expansion Radius of 1 for the main RAIDAL configuration.

`active_learning_modules/rank_by_feature.py` fixes those values when `raidal` is selected. The K and R strategy variants change one of them while keeping the rest of the acquisition procedure unchanged.

RAIDAL first L2 normalizes the BiLSTM features. It then keeps only the features inside the CTC decoder supported Temporal Region of Interest. For each retained feature, the implementation stores the distance to its nearest feature in the current labeled pool. A candidate score is the sum of these distances divided by the raw frame count of the video.

After a video is selected, its retained features become part of the reference set. The code updates the stored nearest neighbor distances against the newly selected video instead of recomputing the distance to the complete labeled pool from scratch.

During selection, RAIDAL prints how much the TROI filter shrank the videos, which is a quick sanity check that decoder-based filtering is actually doing something:

```text
Avg size after TROI filtering: 31.42% +- 8.71% of original video. Q15 = ...
```

If that number is 100%, the filter isn't removing anything and the entire video is influencing selection, so always try to keep it lower. In my experiments, the standard RAIDAL configuration usually retained around 20%-40% of the videos.

## Output

Every labeled iteration gets a work directory. It contains the best model, the labeled and unlabeled annotation files used for checkpointing, evaluation outputs, the saved training configuration, and `ALParams.json` for active learning runs.

Feature extraction creates a separate directory with one NumPy file per video. Every file holds the BiLSTM features, the classifier logits, the decoded CTC sequences and the alignment peak timesteps from a single clean forward pass. When `-b` is above zero, each file additionally holds `-b` extra copies of the classifier logits and `-b` extra copies of the decoded sequences, one of each per dropout pass. The BiLSTM features and the alignment peak timesteps are only ever saved once, from the clean pass, so raising `-b` never duplicates those two.

The acquisition ranking is written to `videoRank.json`. Each acquisition strategy will rank videos differently, so the values between two rankings generated by two different acquisition strategies are not comparable. What *is* comparable across strategies is the order, and which videos ended up in the labeled pool.

`activelearning.py` will automatically prepare the dataset for the next acquisition after every trained iteration. This also happens after the last trained budget in the requested schedule. Because of that, one additional labeled and unlabeled dataset pair can be created after the last evaluated point. That extra pair is only the prepared next state and does not correspond to another trained result.

## Restarting an experiment

Pass `--checkpoint` a labeled iteration work directory containing `labeled.corpus.csv` and `unlabeled.corpus.csv`.

Use `--use-model` when the saved model should be reused but the features should be generated again.

Use `--use-features` when both the initial model and pregenerated feature directories should be reused.

Use a new value for `-n`. The parser will automatically reject experiments with repeated names to avoid accidentally overwriting your own files.

Both `--use-model` and `--use-features` also require `_best_model.pt` to be present in the checkpoint directory, and `--use-features` additionally copies `test.txt` and `dev.txt` from it. Neither flag works without `--checkpoint`.

## Troubleshooting

Most failures happen in the parser before anything expensive starts, and the messages are deliberately specific.

**`All elements in num-frames-to-label list must be greater than 0!`**: every value in `-f` has to be positive, including the first one.

**`... -labeled already exists!`** or **`... -unlabeled already exists!`**: a previous run left dataset copies next to your dataset folder. These are created by the program, so the safe fix is to delete the leftover `dataset/<name>-labeled-iterationN` and `-unlabeled-iterationN` folders from the interrupted run before starting again. Or just give your experiment a different name with `-n`.

**`'features' folder already exists!`**: delete the stray `features` folder in the working directory. It confuses feature extraction if left behind. Or, again, just give your experiment a different name with `-n`.

**`Please use a custom name different from the one in the checkpoint`**: `-n` can't contain the same experiment name as the one in `--checkpoint`. Use a fresh name per method, e.g. `RaidalRep1` branching from a `RandomRep1` checkpoint.

**`Can't use --use-features without enabling --checkpoint`**: both `--use-features` and `--use-model` only make sense when using a checkpoint, so pass `--checkpoint` too.

**`When activated, BALD should run for at least 2 iterations`**: `-b` must be `0` or `>= 2`.

**A `KeyError` on a video ID during selection**: the frame-count file for your dataset is missing an entry. `frameCount/` ships `phoenix2014.json`, `phoenix2014-T.json` and `Isharah.json`, one per supported dataset. A custom dataset needs its own file there with the number of frames in every video.

## Cleanup

I've included a cleanup script called `cleanup.sh` that automatically removes temporary files from the work directory. `activelearning.py` runs it automatically, so you don't need to worry about running it manually.