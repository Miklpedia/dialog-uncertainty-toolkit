# Methodology Notes

## Speech data (prototype)

Short English clips for the notebooks were sourced from [**Common Voice
Spontaneous Speech 3.0 — English**](https://mozilladatacollective.com/datasets/cmn1pv5hi00uto1072y1074y7)
on Mozilla Data Collective: crowdsourced **spontaneous** answers (as opposed to
read sentences). The published license is **CC0-1.0**; the dataset card also
states constraints (e.g. no speaker identification; no re-hosting of the
corpus). Re-check the card if you use a different release or mirror.

The reported exploratory runs use **n = 20** utterances (a convenience subset,
not an official split from the release), roughly **10–30 seconds** each, as
`.mp3` or `.wav` under `data/`. **Raw audio is not
checked into this repository** (see `data/README.md`); pipelines write derived
CSVs and figures under `output/` only on the author’s machine.

This corpus is a pragmatic stand-in for “spoken user input” when prototyping
ASR + prosody + LLM uncertainty. It is **not** full conversational dialogue;
claims should stay scoped to signal extraction and correlation on this sample.

## ASR Confidence (Whisper)

Whisper outputs token-level log-probabilities for each transcribed token.
We aggregate them per utterance using the mean log-probability over all
generated tokens (`avg_logprob` in Whisper's segment output).

**Caveats**:
- Mean log-prob is not calibrated; lower values indicate higher model
  uncertainty but the absolute scale depends on input length and
  acoustic conditions.
- Whisper-medium is used by default; for very long audio, consider
  whisper-large-v3 if VRAM allows.

## LLM Token Entropy

For each generation step, we compute Shannon entropy over the next-token
probability distribution. We report the mean and max across the
generated sequence.

## Semantic Entropy (Simplified Kuhn et al., 2023)

Original method: sample N responses, cluster them using bidirectional
NLI entailment, compute entropy over cluster sizes.

**This implementation**: replaces the NLI clustering with cosine-similarity
agglomerative clustering on Sentence-BERT embeddings. This is a simpler
approximation; expect slightly different absolute values from the original
paper but similar ranking behavior across prompts.

## Prosodic Features

Extracted using Parselmouth (Python wrapper around Praat):

- **F0 (pitch)**: mean, std, range over voiced frames; pitch floor 75 Hz,
  ceiling 600 Hz.
- **Intensity (loudness)**: mean and std in dB over the full utterance.
- **Voiced fraction**: proportion of frames with detected F0.
- **Pause ratio**: proportion of frames below `mean - 1.5σ` intensity
  threshold (rough silence detector).
- **Speaking rate**: voiced frames per second (proxy; not phoneme-rate).

## Cross-Signal Analysis

The combined notebook computes Pearson correlation across all numerical
signals on the small sample. This is exploratory; significance testing
is not appropriate at this sample size.

## Future Work

- Expand to a labeled subset (response-error labels via LLM-as-Judge
  + human calibration on ~200 samples).
- Train a learning-to-defer policy using these signals as features
  (Mozannar & Sontag, 2020).
- Evaluate on speech-native dialogue corpora (e.g., Spoken-WOZ, DSTC).