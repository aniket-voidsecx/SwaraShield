import numpy as np
import onnxruntime as ort
import soundfile as sf


AUDIO_PATH = "data/sample_audio/genuine_voice.wav"
MODEL_PATH = "models/aasist/aasist.onnx"

TARGET_SAMPLE_RATE = 16000
TARGET_LENGTH = 64600


print("================================")
print(" VoiceSentinel AASIST Test")
print("================================")


# ------------------------------------------------------------
# LOAD AUDIO
# ------------------------------------------------------------

print("\nLoading audio...")

audio, sample_rate = sf.read(
    AUDIO_PATH,
    dtype="float32"
)

print("Original sample rate:", sample_rate)
print("Original shape:", audio.shape)


# ------------------------------------------------------------
# MONO
# ------------------------------------------------------------

if audio.ndim > 1:

    print("Converting stereo → mono...")

    audio = np.mean(
        audio,
        axis=1
    )


# ------------------------------------------------------------
# RESAMPLE
# ------------------------------------------------------------

if sample_rate != TARGET_SAMPLE_RATE:

    print(
        f"Resampling {sample_rate} Hz → "
        f"{TARGET_SAMPLE_RATE} Hz..."
    )

    from scipy.signal import resample_poly

    audio = resample_poly(
        audio,
        TARGET_SAMPLE_RATE,
        sample_rate
    ).astype(np.float32)

else:

    print("Sample rate already 16 kHz.")


# ------------------------------------------------------------
# NORMALIZE
# ------------------------------------------------------------

audio = audio.astype(np.float32)

peak = np.max(np.abs(audio))

if peak > 0:

    audio = audio / peak


# ------------------------------------------------------------
# EXACT AASIST INPUT LENGTH
# ------------------------------------------------------------

if len(audio) < TARGET_LENGTH:

    print(
        f"Padding audio: {len(audio)} → {TARGET_LENGTH}"
    )

    audio = np.pad(
        audio,
        (0, TARGET_LENGTH - len(audio))
    )

else:

    print(
        f"Trimming audio: {len(audio)} → {TARGET_LENGTH}"
    )

    audio = audio[:TARGET_LENGTH]


# ------------------------------------------------------------
# BATCH
# ------------------------------------------------------------

audio = audio.reshape(
    1,
    TARGET_LENGTH
)

audio = np.ascontiguousarray(
    audio,
    dtype=np.float32
)

print(
    "Final model input:",
    audio.shape
)


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("\nLoading AASIST ONNX model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

print("AASIST model loaded.")


# ------------------------------------------------------------
# INFERENCE
# ------------------------------------------------------------

print("\nRunning inference...")

outputs = session.run(
    None,
    {
        "wav": audio
    }
)

logits = outputs[0]

print("Raw logits:", logits)


# ------------------------------------------------------------
# SOFTMAX
# ------------------------------------------------------------

exp_values = np.exp(
    logits - np.max(
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
spoof_probability = class_0
bonafide_probability = class_1

if spoof_probability >= bonafide_probability:
    verdict = "SPOOF / AI-GENERATED"
    confidence = spoof_probability
else:
    verdict = "BONAFIDE / HUMAN"
    confidence = bonafide_probability

# ------------------------------------------------------------
# RESULT
# ------------------------------------------------------------
print(
    f"Spoof probability    : {spoof_probability:.2f}%"
)

print(
    f"Bonafide probability : {bonafide_probability:.2f}%"
)

print(
    f"\nVerdict              : {verdict}"
)

print(
    f"Model confidence     : {confidence:.2f}%"
)
print("\n================================")
print(" AASIST RESULT")
print("================================")

print(
    f"Class 0 probability : {class_0:.2f}%"
)

print(
    f"Class 1 probability : {class_1:.2f}%"
)

print("\nInference completed successfully.")

print("\nNOTE:")
print("Class 0/1 mapping will be verified separately.")