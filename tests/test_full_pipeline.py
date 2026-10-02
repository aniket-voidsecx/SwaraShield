import numpy as np
import onnxruntime as ort
import soundfile as sf

from modules.risk_engine import calculate_voice_risk


# ============================================================
# SETTINGS
# ============================================================

AUDIO_PATH = "data/sample_audio/genuine_voice.wav"
MODEL_PATH = "models/aasist/aasist.onnx"

TARGET_SAMPLE_RATE = 16000
TARGET_LENGTH = 64600


print("================================")
print(" VoiceSentinel Full AI Pipeline")
print("================================")


# ============================================================
# LOAD AUDIO
# ============================================================

print("\n[1/5] Loading audio...")

audio, sample_rate = sf.read(
    AUDIO_PATH,
    dtype="float32"
)

print("Sample rate:", sample_rate)
print("Audio shape:", audio.shape)


# ============================================================
# MONO
# ============================================================

if audio.ndim > 1:

    audio = np.mean(
        audio,
        axis=1
    )


# ============================================================
# RESAMPLE
# ============================================================

if sample_rate != TARGET_SAMPLE_RATE:

    print(
        "Resampling audio..."
    )

    from scipy.signal import resample_poly

    audio = resample_poly(
        audio,
        TARGET_SAMPLE_RATE,
        sample_rate
    ).astype(np.float32)


# ============================================================
# NORMALIZE
# ============================================================

audio = audio.astype(
    np.float32
)

peak = np.max(
    np.abs(audio)
)

if peak > 0:

    audio = audio / peak


# ============================================================
# FIX LENGTH
# ============================================================

if len(audio) < TARGET_LENGTH:

    audio = np.pad(
        audio,
        (
            0,
            TARGET_LENGTH - len(audio)
        )
    )

else:

    audio = audio[
        :TARGET_LENGTH
    ]


# ============================================================
# BATCH
# ============================================================

audio = audio.reshape(
    1,
    TARGET_LENGTH
)

audio = np.ascontiguousarray(
    audio,
    dtype=np.float32
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n[2/5] Loading AASIST...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=[
        "CPUExecutionProvider"
    ]
)

print("AASIST loaded.")


# ============================================================
# INFERENCE
# ============================================================

print("\n[3/5] Running AI voice detection...")

outputs = session.run(
    None,
    {
        "wav": audio
    }
)

logits = outputs[0]

print(
    "Raw logits:",
    logits
)


# ============================================================
# SOFTMAX
# ============================================================

exp_values = np.exp(
    logits -
    np.max(
        logits,
        axis=1,
        keepdims=True
    )
)

probabilities = (
    exp_values /
    np.sum(
        exp_values,
        axis=1,
        keepdims=True
    )
)


class_0 = float(
    probabilities[0][0]
) * 100

class_1 = float(
    probabilities[0][1]
) * 100


# ============================================================
# TEMPORARY MODEL MAPPING
# ============================================================

# IMPORTANT:
# This mapping is temporary until we validate the exact
# AASIST checkpoint's class convention.

spoof_probability = class_0


# ============================================================
# RISK ENGINE
# ============================================================

print("\n[4/5] Calculating risk...")

risk = calculate_voice_risk(
    spoof_probability
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n[5/5] Final VoiceSentinel Result")

print("--------------------------------")
print(
    f"Class 0 probability : {class_0:.2f}%"
)

print(
    f"Class 1 probability : {class_1:.2f}%"
)

print(
    f"Spoof probability   : "
    f"{spoof_probability:.2f}%"
)

print(
    f"Risk Score          : "
    f"{risk['risk_score']:.2f}/100"
)

print(
    f"Risk Level          : "
    f"{risk['risk_level']}"
)

print(
    f"Recommendation      : "
    f"{risk['recommendation']}"
)

print("--------------------------------")

print("\nFull pipeline completed successfully.")