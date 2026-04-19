# Methodology Notes

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