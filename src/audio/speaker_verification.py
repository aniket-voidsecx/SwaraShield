"""
SwaraShield Speaker Verification
ECAPA-TDNN based speaker verification

Features:
- Registered trusted voice reference
- File-to-file speaker verification
- Live audio chunk verification
- Safe Path/string handling
"""

import os
import tempfile
import wave

import numpy as np
from speechbrain.inference.speaker import SpeakerRecognition


# ============================================================
# GLOBAL MODEL
# ============================================================

_model = None


# ============================================================
# LOAD ECAPA MODEL
# ============================================================

def load_speaker_model():
    global _model

    if _model is None:
        print("[ECAPA] Loading speaker verification model...")

        _model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/speaker_ecapa",
        )

        print("[ECAPA] Model loaded successfully.")

    return _model


# ============================================================
# NORMALIZE PATH
# ============================================================

def _path_string(path):
    """
    Convert pathlib.Path / string into a normal string path.
    """
    return os.fspath(path)


# ============================================================
# FILE-TO-FILE VERIFICATION
# ============================================================

def verify_speaker(
    registered_voice_path,
    current_voice_path,
):
    """
    Compare registered trusted voice with current voice.

    Returns:
        similarity
        status
        risk
        raw_score
    """

    registered_voice_path = _path_string(
        registered_voice_path
    )

    current_voice_path = _path_string(
        current_voice_path
    )

    model = load_speaker_model()

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(registered_voice_path):
        raise FileNotFoundError(
            f"Registered voice not found: "
            f"{registered_voice_path}"
        )

    if not os.path.exists(current_voice_path):
        raise FileNotFoundError(
            f"Current voice not found: "
            f"{current_voice_path}"
        )

    # --------------------------------------------------------
    # ECAPA verification
    # --------------------------------------------------------

    score, prediction = model.verify_files(
        registered_voice_path,
        current_voice_path,
    )

    raw_score = float(score)

    # SpeechBrain similarity is normally in approximately
    # 0-1 range for this model.
    similarity = raw_score * 100.0

    similarity = max(
        0.0,
        min(
            similarity,
            100.0,
        ),
    )

    # --------------------------------------------------------
    # Speaker risk
    # --------------------------------------------------------

    if similarity >= 85.0:

        status = "CONSISTENT"
        risk = "LOW"

    elif similarity >= 65.0:

        status = "REVIEW REQUIRED"
        risk = "MEDIUM"

    else:

        status = "SPEAKER MISMATCH"
        risk = "HIGH"

    result = {
        "similarity": round(
            similarity,
            2,
        ),
        "status": status,
        "risk": risk,
        "raw_score": round(
            raw_score,
            4,
        ),
        "same_speaker": bool(prediction),
    }

    return result


# ============================================================
# SPEAKER RISK SCORE
# ============================================================

def get_speaker_risk_score(
    registered_voice_path,
    current_voice_path,
):
    """
    Convert ECAPA similarity into speaker risk.

    High similarity  -> low risk
    Low similarity   -> high risk
    """

    result = verify_speaker(
        registered_voice_path,
        current_voice_path,
    )

    speaker_risk = (
        100.0
        - result["similarity"]
    )

    speaker_risk = max(
        0.0,
        min(
            speaker_risk,
            100.0,
        ),
    )

    return round(
        speaker_risk,
        2,
    )


# ============================================================
# LIVE CHUNK VERIFICATION
# ============================================================

def verify_live_chunk(
    registered_voice_path,
    live_audio,
    sample_rate=16000,
):
    """
    Compare a LiveKit audio chunk against
    the registered trusted voice.

    Parameters
    ----------
    registered_voice_path:
        Path to genuine_voice.wav

    live_audio:
        numpy float32 mono audio
        approximately -1.0 to +1.0

    sample_rate:
        Expected sample rate.
        Default = 16000 Hz

    Returns
    -------
    dict
        Same format as verify_speaker()
    """

    # --------------------------------------------------------
    # Normalize registered path
    # --------------------------------------------------------

    registered_voice_path = _path_string(
        registered_voice_path
    )

    # --------------------------------------------------------
    # Validate registered voice
    # --------------------------------------------------------

    if not os.path.exists(
        registered_voice_path
    ):
        raise FileNotFoundError(
            "Trusted registered voice not found: "
            + registered_voice_path
        )

    # --------------------------------------------------------
    # Convert live audio to numpy
    # --------------------------------------------------------

    live_audio = np.asarray(
        live_audio,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Flatten audio
    # --------------------------------------------------------

    live_audio = live_audio.reshape(-1)

    # --------------------------------------------------------
    # Prevent invalid values
    # --------------------------------------------------------

    live_audio = np.nan_to_num(
        live_audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    live_audio = np.clip(
        live_audio,
        -1.0,
        1.0,
    )

    # --------------------------------------------------------
    # Need enough audio
    # --------------------------------------------------------

    if len(live_audio) < 1600:
        raise ValueError(
            "Live audio chunk is too short for "
            "speaker verification."
        )

    # --------------------------------------------------------
    # Convert float32 -> PCM16
    # --------------------------------------------------------

    pcm16 = (
        live_audio * 32767.0
    ).astype(
        np.int16
    )

    temp_path = None

    try:

        # ----------------------------------------------------
        # Create temporary WAV
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temp_file:

            temp_path = temp_file.name

        # ----------------------------------------------------
        # Write WAV
        # ----------------------------------------------------

        with wave.open(
            temp_path,
            "wb",
        ) as wav_file:

            wav_file.setnchannels(1)

            wav_file.setsampwidth(2)

            wav_file.setframerate(
                sample_rate
            )

            wav_file.writeframes(
                pcm16.tobytes()
            )

        # ----------------------------------------------------
        # Run ECAPA
        # ----------------------------------------------------

        result = verify_speaker(
            registered_voice_path,
            temp_path,
        )

        return result

    finally:

        # ----------------------------------------------------
        # Delete temporary WAV
        # ----------------------------------------------------

        if (
            temp_path
            and os.path.exists(temp_path)
        ):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ============================================================
# LIVE SPEAKER RISK
# ============================================================

def get_live_speaker_risk(
    registered_voice_path,
    live_audio,
    sample_rate=16000,
):
    """
    Convenience function for LiveKit pipeline.

    Returns speaker risk from 0-100.
    """

    result = verify_live_chunk(
        registered_voice_path,
        live_audio,
        sample_rate,
    )

    risk = (
        100.0
        - result["similarity"]
    )

    risk = max(
        0.0,
        min(
            risk,
            100.0,
        ),
    )

    return round(
        risk,
        2,
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    registered = (
        "data/sample_audio/genuine_voice.wav"
    )

    print("=" * 60)
    print("SwaraShield ECAPA Speaker Verification Test")
    print("=" * 60)

    print(
        f"Trusted voice: {registered}"
    )

    if not os.path.exists(registered):

        print(
            "ERROR: genuine_voice.wav not found."
        )

        raise SystemExit(1)

    print(
        "Testing trusted voice against itself..."
    )

    result = verify_speaker(
        registered,
        registered,
    )

    print()
    print(
        "Similarity :",
        result["similarity"],
        "%",
    )

    print(
        "Status     :",
        result["status"],
    )

    print(
        "Risk       :",
        result["risk"],
    )

    print(
        "Same speaker:",
        result["same_speaker"],
    )

    print("=" * 60)