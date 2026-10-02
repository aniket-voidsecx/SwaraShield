# Models

This directory contains model-related resources and configuration.

## AASIST

SwaraShield currently uses/experiments with AASIST for audio anti-spoofing.

Before redistributing any model file, verify the original model's:

- License
- Redistribution terms
- Attribution requirements
- Source
- Version
- Integrity/checksum

If redistribution is not permitted, do not commit the model weights. Instead, document how contributors can obtain them from the authorized source.

## Recommended structure

```text
models/
└── aasist/
    ├── README.md
    └── aasist.onnx
```

Large model files may be better handled through an approved model registry or release artifact rather than normal Git history.
