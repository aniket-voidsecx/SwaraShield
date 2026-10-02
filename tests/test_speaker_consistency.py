import numpy as np
import soundfile as sf

from speaker_consistency import (
    get_speaker_consistency_result
)


def load_audio(file_path, target_length=64000):
    print(f"Loading: {file_path}")

    audio, sample_rate = sf.read(file_path)

    # Stereo → Mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    print(f"Sample rate: {sample_rate}")
    print(f"Original shape: {audio.shape}")

    # Same fixed length for both samples
    if len(audio) > target_length:
        audio = audio[:target_length]

    elif len(audio) < target_length:
        padding = target_length - len(audio)

        audio = np.pad(
            audio,
            (0, padding),
            mode="constant"
        )

    print(f"Final shape: {audio.shape}")

    return audio


print("=" * 55)
print(" VoiceSentinel Speaker Consistency Test")
print("=" * 55)

# --------------------------------------------------
# REGISTERED GENUINE VOICE
# --------------------------------------------------

registered_voice = load_audio(
    "data/sample_audio/genuine_voice.wav"
)

print("\n")

# --------------------------------------------------
# CURRENT VOICE TO VERIFY
# --------------------------------------------------

current_voice = load_audio(
    "data/sample_audio/test_pcm.wav"
)

print("\nRunning speaker consistency check...\n")

result = get_speaker_consistency_result(
    registered_voice,
    current_voice
)

print("=" * 55)
print(" SPEAKER CONSISTENCY RESULT")
print("=" * 55)

print(
    f"Similarity Score : "
    f"{result['similarity']}%"
)

print(
    f"Speaker Status   : "
    f"{result['status']}"
)

print(
    f"Risk Level       : "
    f"{result['risk']}"
)

print("=" * 55)
print("\nTest completed successfully.")