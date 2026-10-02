# SwaraShield Architecture

## Current prototype

```text
Browser Microphone
       |
       v
LiveKit / WebRTC
       |
       v
Audio Interceptor
       |
       v
Audio Preprocessing
       |
       v
AASIST Anti-Spoof Model
       |
       v
Voice Risk
       |
       +-------------------+
       |                   |
       v                   v
Speaker Verification   Conversation Analysis
       |                   |
       +---------+---------+
                 |
                 v
            Risk Engine
                 |
                 v
          Security Status
```

## Audio pipeline

The prototype targets:

```text
Input → mono → normalized → resampled → fixed inference window → model
```

Current target sample rate:

```text
16000 Hz
```

The implementation should keep audio preprocessing deterministic and testable.

## Risk engine

The risk engine should not blindly trust a single model.

A future implementation can combine:

```text
Voice spoof signal
Conversation risk
Speaker verification
Behavioral signals
Threat intelligence
        ↓
   Risk fusion
        ↓
Final risk
```

## Design principles

1. Modular components
2. Testable preprocessing
3. Explicit thresholds
4. Explainable risk signals
5. Privacy by design
6. Low-latency processing
7. Graceful degradation
8. No secret material in source code

## Future architecture

The long-term system may include separate services for:

- Audio ingestion
- Speech-to-text
- Voice authenticity
- Speaker verification
- Fraud/NLP detection
- Threat intelligence
- Risk fusion
- User notification
- Mobile client
