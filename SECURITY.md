# Security Policy

## Scope

SwaraShield is a research/prototype project involving audio processing, AI models, real-time communications, and security-sensitive use cases.

## Please do not

Do not report secrets, API keys, credentials, private recordings, or personal data in public GitHub issues.

Do not test the system against communications you do not own or have permission to analyze.

## Reporting a vulnerability

If you discover a vulnerability that could expose credentials, private data, remote systems, or users, please report it privately to the repository maintainers rather than opening a public issue.

Until a dedicated security contact is configured, use GitHub's private vulnerability reporting feature if it is enabled for the repository.

Include:

- A clear description
- Affected component/version
- Reproduction steps
- Security impact
- Suggested mitigation, if known
- Any relevant logs that do not contain secrets or personal data

## Secrets

Never commit:

```text
.env
API keys
LiveKit API secrets
private certificates
passwords
tokens
private datasets
private call recordings
```

If a secret is accidentally committed, rotate/revoke it immediately. Removing it from Git history does not make the secret safe.

## Responsible disclosure

We encourage responsible disclosure and coordinated remediation.

Please allow maintainers reasonable time to investigate before public disclosure.
