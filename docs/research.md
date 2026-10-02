# SwaraShield Research Notes

## Research goal

Build a reliable, low-latency, privacy-conscious system for detecting suspicious voice and conversation signals during controlled real-time communications.

## Key research questions

### Voice spoof detection

- How stable is the model across consecutive chunks?
- How does background noise affect performance?
- How does compression affect predictions?
- How does performance vary by speaker?
- How does performance change across languages?
- What threshold produces acceptable false-positive/false-negative tradeoffs?

### Risk fusion

A risk score should combine multiple independent signals rather than treating one classifier as ground truth.

Potential signals:

```text
spoof_score
speaker_verification_score
fraud_score
behavioral_score
threat_intelligence_score
```

## Metrics

Recommended metrics include:

- False Positive Rate
- False Negative Rate
- Precision
- Recall
- F1
- ROC-AUC where applicable
- EER for speaker/anti-spoof tasks where appropriate
- Calibration error
- Median detection latency
- P95 detection latency
- Resource consumption

## Reproducibility

Every experiment should record:

- Model version
- Dataset/version
- Dataset license
- Sampling rate
- Preprocessing
- Threshold
- Hardware
- Software versions
- Random seed when applicable
- Evaluation metrics

## Important warning

A high score on a small sample is not evidence of production readiness.

SwaraShield should be evaluated across realistic conditions before user-facing security decisions are enabled.
