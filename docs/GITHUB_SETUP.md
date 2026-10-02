# GitHub Setup Checklist

## Before first push

- [ ] Review `.gitignore`
- [ ] Remove `.env`
- [ ] Remove API keys/secrets
- [ ] Remove private recordings
- [ ] Check model license
- [ ] Add screenshots
- [ ] Test installation
- [ ] Run `pytest`
- [ ] Run `python -m compileall src app`
- [ ] Review Git diff

## Create repository

Recommended:

```text
Repository: SwaraShield
Description:
AI-powered open-source caller protection against voice spoofing, scams and social engineering.
```

Suggested topics:

```text
ai
cybersecurity
voice-security
voice-cloning
deepfake-detection
scam-detection
fraud-detection
machine-learning
audio-ml
webrtc
livekit
python
opensource
```

## First release

Use:

```text
v0.1.0
```

Release title:

```text
SwaraShield v0.1.0 — Research Prototype
```

## Suggested branch model

```text
main
 ├── develop
 ├── feature/*
 ├── fix/*
 └── research/*
```

For a small project, `main + feature/*` is also sufficient.

## Useful repository settings

Enable:

- Issues
- Discussions
- Pull Requests
- Security advisories / private vulnerability reporting if available

Add labels:

```text
bug
enhancement
research
good first issue
help wanted
documentation
security
machine learning
audio
mobile
```
