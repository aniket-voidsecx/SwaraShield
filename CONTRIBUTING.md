# Contributing to SwaraShield

Thank you for your interest in contributing to SwaraShield! 🛡️

SwaraShield is an open-source project focused on AI-assisted caller security, voice-spoof detection, scam detection, and defensive cybersecurity.

## Ways to contribute

You can contribute through:

- Code
- Machine-learning experiments
- Audio processing improvements
- Model evaluation
- Dataset documentation
- Security research
- UI/UX
- Documentation
- Testing
- Bug reports
- Feature proposals

## Before you start

1. Read the README.
2. Check existing Issues and Discussions.
3. Avoid duplicating an existing issue.
4. Never upload secrets, credentials, private call recordings, or personal data.
5. For security vulnerabilities, follow `SECURITY.md` instead of opening a public issue.

## Development workflow

```bash
git clone https://github.com/YOUR_USERNAME/SwaraShield.git
cd SwaraShield

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Create a feature branch:

```bash
git checkout -b feature/your-feature
```

Make your changes, then run:

```bash
pytest
```

Before opening a PR, also check:

```bash
python -m compileall src app
```

Commit:

```bash
git add .
git commit -m "feat: describe your change"
```

Push:

```bash
git push origin feature/your-feature
```

## Pull requests

A good PR should include:

- What changed
- Why it changed
- How it was tested
- Any limitations
- Screenshots for UI changes
- Evaluation results for ML changes

Keep PRs focused. A small, understandable PR is easier to review.

## ML / research contributions

If you change a model or detection method, document:

- Dataset/source
- Dataset license
- Preprocessing
- Train/validation/test split
- Model/version
- Metrics
- Thresholds
- Hardware
- Latency
- Known failure cases

Do not present a model as production-ready based on a small demo.

## Audio and privacy

Only contribute audio that you have the legal right to use.

Do not upload:

- Private phone calls
- Personal recordings
- Identifiable speech without appropriate consent
- Sensitive financial information
- OTPs or authentication information

## Commit style

Recommended prefixes:

```text
feat:
fix:
docs:
test:
refactor:
perf:
security:
chore:
```

Example:

```text
feat: add rolling spoof-risk smoothing
```

## Code of conduct

By participating, you agree to follow `CODE_OF_CONDUCT.md`.

Thank you for helping build SwaraShield. 🚀
