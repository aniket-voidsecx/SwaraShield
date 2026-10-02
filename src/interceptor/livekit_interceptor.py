"""
SwaraShield LiveKit Interceptor
--------------------------------

Live pipeline:

LiveKit audio
    ↓
4-second audio chunk
    ├── AASIST      → voice spoof / clone risk
    ├── ECAPA       → speaker consistency risk
    └── Whisper     → live transcript
                         ↓
                    Fraud Detector
                         ↓
                     fraud risk
    ↓
Risk Fusion
    ↓
Security Decision
    ↓
ALLOW / WARNING / VERIFICATION / TRANSACTION FROZEN
"""

import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from dotenv import load_dotenv
from livekit import api, rtc

from faster_whisper import WhisperModel

from fraud_detector import detect_fraud_patterns
from modules.risk_engine import calculate_voice_risk
from speaker_verification import verify_live_chunk


# ============================================================
# PROJECT / ENVIRONMENT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.getenv(
    "SWARASHIELD_AASIST_MODEL",
    str(
        PROJECT_ROOT
        / "models"
        / "aasist"
        / "aasist.onnx"
    ),
)

ROOM_NAME = os.getenv(
    "LIVEKIT_ROOM",
    "swarashield-demo",
).strip()

TARGET_SR = 16000

TARGET_LENGTH = 64600


# ------------------------------------------------------------
# 4-second analysis chunks
# ------------------------------------------------------------

CHUNK_SECONDS = 4.0

CHUNK_SAMPLES = int(
    TARGET_SR * CHUNK_SECONDS
)


# ------------------------------------------------------------
# AASIST smoothing
# ------------------------------------------------------------

SMOOTHING_WINDOW = 3


# ------------------------------------------------------------
# Whisper
# ------------------------------------------------------------

WHISPER_MODEL_SIZE = "tiny"

whisper_model = None


# ------------------------------------------------------------
# Dashboard status
# ------------------------------------------------------------

STATUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "live_status.json"
)


# ------------------------------------------------------------
# Trusted speaker reference
# ------------------------------------------------------------

REGISTERED_VOICE_PATH = (
    PROJECT_ROOT
    / "data"
    / "sample_audio"
    / "genuine_voice.wav"
)


# ============================================================
# STATUS WRITER
# ============================================================

def write_status(payload):

    STATUS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        STATUS_PATH.with_suffix(".tmp")
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(
        STATUS_PATH
    )


# ============================================================
# LIVEKIT WORKER TOKEN
# ============================================================

def make_worker_token():

    key = os.getenv(
        "LIVEKIT_API_KEY",
        "",
    ).strip()

    secret = os.getenv(
        "LIVEKIT_API_SECRET",
        "",
    ).strip()

    if not key or not secret:

        raise RuntimeError(
            "LIVEKIT_API_KEY / "
            "LIVEKIT_API_SECRET are required."
        )

    grants = api.VideoGrants(

        room_join=True,

        room=ROOM_NAME,

        can_publish=False,

        can_subscribe=True,

        can_publish_data=False,
    )

    return (
        api.AccessToken(
            key,
            secret,
        )
        .with_identity(
            "swarashield-interceptor"
        )
        .with_grants(
            grants
        )
        .to_jwt()
    )


# ============================================================
# AUDIO CONVERSION
# ============================================================

def resample_mono_int16(frame):

    data = np.asarray(
        frame.data,
        dtype=np.int16,
    ).astype(np.float32)

    # --------------------------------------------------------
    # Stereo / multichannel → mono
    # --------------------------------------------------------

    if frame.num_channels > 1:

        data = data.reshape(
            -1,
            frame.num_channels,
        ).mean(axis=1)

    # --------------------------------------------------------
    # int16 → float32 [-1, 1]
    # --------------------------------------------------------

    data /= 32768.0

    # --------------------------------------------------------
    # Resample → 16 kHz
    # --------------------------------------------------------

    if frame.sample_rate != TARGET_SR:

        new_len = max(
            1,
            int(
                round(
                    len(data)
                    * TARGET_SR
                    / frame.sample_rate
                )
            ),
        )

        old_x = np.linspace(
            0.0,
            1.0,
            len(data),
            endpoint=False,
        )

        new_x = np.linspace(
            0.0,
            1.0,
            new_len,
            endpoint=False,
        )

        data = np.interp(
            new_x,
            old_x,
            data,
        ).astype(np.float32)

    return data


# ============================================================
# AASIST AUDIO PREPARATION
# ============================================================

def prepare_aasist_audio(audio):

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    audio = np.clip(
        audio,
        -1.0,
        1.0,
    )

    if len(audio) < TARGET_LENGTH:

        audio = np.pad(
            audio,
            (
                0,
                TARGET_LENGTH - len(audio),
            ),
        )

    return audio[:TARGET_LENGTH]


# ============================================================
# AASIST DETECTION
# ============================================================

def detect_voice(
    session,
    audio,
):

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    model_input = (
        prepare_aasist_audio(
            audio
        )
        .reshape(
            1,
            TARGET_LENGTH,
        )
    )

    outputs = session.run(
        None,
        {
            input_name: model_input
        },
    )

    logits = outputs[0]

    # --------------------------------------------------------
    # Stable softmax
    # --------------------------------------------------------

    exp_logits = np.exp(
        logits
        - np.max(
            logits,
            axis=1,
            keepdims=True,
        )
    )

    probabilities = (
        exp_logits
        /
        np.sum(
            exp_logits,
            axis=1,
            keepdims=True,
        )
    )

    # --------------------------------------------------------
    # SwaraShield interpretation
    #
    # class 0 = spoof
    # class 1 = genuine
    # --------------------------------------------------------

    spoof_probability = float(
        probabilities[0][0]
    )

    genuine_probability = float(
        probabilities[0][1]
    )

    return (
        spoof_probability,
        genuine_probability,
    )


# ============================================================
# WHISPER MODEL
# ============================================================

def load_whisper_model():

    global whisper_model

    if whisper_model is None:

        print(
            "[WHISPER] "
            "Loading speech-to-text model..."
        )

        whisper_model = WhisperModel(

            WHISPER_MODEL_SIZE,

            device="cpu",

            compute_type="int8",
        )

        print(
            "[WHISPER] "
            "Model loaded successfully."
        )

    return whisper_model


# ============================================================
# LIVE TRANSCRIPTION — MULTILINGUAL
# ============================================================

def transcribe_live_audio(
    audio,
):
    whisper = load_whisper_model()

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    audio = np.clip(
        audio,
        -1.0,
        1.0,
    )

    # --------------------------------------------------------
    # Automatic language detection
    # --------------------------------------------------------

    segments, info = whisper.transcribe(
        audio,
        language=None,
        vad_filter=False,
        beam_size=5,
        condition_on_previous_text=False,
        temperature=0,
    )

    detected_language = getattr(
        info,
        "language",
        "unknown",
    )

    transcript_parts = []

    for segment in segments:
        text = segment.text.strip()

        if text:
            transcript_parts.append(text)

    transcript = " ".join(
        transcript_parts
    ).strip()

    print(
        f"[WHISPER] Language={detected_language} "
        f"| Transcript={transcript}"
    )

    return transcript


# ============================================================
# RISK FUSION
# ============================================================

def fuse_all_risks(
    acoustic_risk,
    speaker_risk,
    fraud_risk,
):

    acoustic_risk = max(
        0.0,
        min(
            100.0,
            float(acoustic_risk),
        ),
    )

    speaker_risk = max(
        0.0,
        min(
            100.0,
            float(speaker_risk),
        ),
    )

    fraud_risk = max(
        0.0,
        min(
            100.0,
            float(fraud_risk),
        ),
    )

    # --------------------------------------------------------
    # Multi-signal fusion
    #
    # AASIST = 60%
    # ECAPA  = 20%
    # Fraud  = 20%
    # --------------------------------------------------------

    fused_risk = (
        0.60 * acoustic_risk
        + 0.20 * speaker_risk
        + 0.20 * fraud_risk
    )

    return round(
        min(
            100.0,
            fused_risk,
        ),
        2,
    )


# ============================================================
# PROTECTION ACTION
# ============================================================

def get_protection_action(
    status,
):

    if status == "CRITICAL":

        return (
            "TRANSACTION_FROZEN",
            True,
        )

    if status == "HIGH":

        return (
            "VERIFICATION_REQUIRED",
            True,
        )

    if status == "MEDIUM":

        return (
            "WARNING",
            True,
        )

    return (
        "ALLOW",
        False,
    )


# ============================================================
# PROCESS LIVE AUDIO TRACK
# ============================================================

async def process_track(
    track,
    participant_identity,
    session,
):

    print(
        f"[AUDIO] Starting analysis for "
        f"{participant_identity}"
    )

    audio_stream = (
        rtc.AudioStream.from_track(

            track=track,

            sample_rate=48000,

            num_channels=1,

            frame_size_ms=100,
        )
    )

    # --------------------------------------------------------
    # Audio buffer
    # --------------------------------------------------------

    buffer = np.empty(
        0,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # AASIST risk history
    # --------------------------------------------------------
    risk_history = []
    # --------------------------------------------------------
    # Protection latch
    # --------------------------------------------------------
    transaction_frozen = False
    critical_latched = False
    critical_snapshot = None
    # --------------------------------------------------------
    # Trusted speaker
    # --------------------------------------------------------

    speaker_reference_available = (
        REGISTERED_VOICE_PATH.exists()
    )

    if speaker_reference_available:

        print(
            "[ECAPA] Trusted reference voice found:"
        )

        print(
            f"[ECAPA] "
            f"{REGISTERED_VOICE_PATH}"
        )

    else:

        print(
            "[ECAPA] WARNING: "
            "genuine_voice.wav not found."
        )

        print(
            "[ECAPA] Speaker verification "
            "will be skipped."
        )

    try:

        async for event in audio_stream:

            # =================================================
            # LIVEKIT FRAME → 16 kHz MONO
            # =================================================

            frame_audio = (
                resample_mono_int16(
                    event.frame
                )
            )

            # =================================================
            # ADD TO BUFFER
            # =================================================

            if len(frame_audio) > 0:

                buffer = np.concatenate(
                    (
                        buffer,
                        frame_audio,
                    )
                )

            # =================================================
            # PROCESS EVERY 4-SECOND CHUNK
            # =================================================

            while len(buffer) >= CHUNK_SAMPLES:

                chunk = buffer[
                    :CHUNK_SAMPLES
                ]

                buffer = buffer[
                    CHUNK_SAMPLES:
                ]

                # =================================================
                # AASIST
                # =================================================

                try:

                    spoof, genuine = (
                        detect_voice(
                            session,
                            chunk,
                        )
                    )

                except Exception as exc:

                    print(
                        "[AASIST ERROR] "
                        f"{participant_identity}: "
                        f"{exc}"
                    )

                    continue

                # =================================================
                # AASIST RISK
                # =================================================

                raw_acoustic_risk = (
                    spoof * 100.0
                )

                # =================================================
                # TEMPORAL SMOOTHING
                # =================================================

                risk_history.append(
                    raw_acoustic_risk
                )

                if (
                    len(risk_history)
                    > SMOOTHING_WINDOW
                ):

                    risk_history.pop(0)

                if len(risk_history) == 1:

                    acoustic_risk = (
                        raw_acoustic_risk
                    )

                else:

                    acoustic_risk = (
                        0.85
                        * raw_acoustic_risk
                        +
                        0.15
                        * risk_history[-2]
                    )

                acoustic_risk = round(
                    acoustic_risk,
                    2,
                )

                # =================================================
                # WHISPER + FRAUD DETECTION
                # =================================================

                transcript = ""

                fraud_risk = 0.0

                fraud_alert = "LOW"

                fraud_patterns = []

                fraud_keywords = []

                try:

                    transcript = (
                        transcribe_live_audio(
                            chunk
                        )
                    )

                    if transcript:

                        fraud_result = (
                            detect_fraud_patterns(
                                transcript
                            )
                        )

                        fraud_risk = float(
                            fraud_result[
                                "fraud_risk"
                            ]
                        )

                        fraud_alert = str(
                            fraud_result[
                                "alert"
                            ]
                        )

                        fraud_patterns = (
                            fraud_result[
                                "matched_patterns"
                            ]
                        )

                        fraud_keywords = (
                            fraud_result[
                                "matched_keywords"
                            ]
                        )

                        print(
                            f"[FRAUD] "
                            f"{participant_identity} | "
                            f"Risk="
                            f"{fraud_risk:.2f}% | "
                            f"Alert="
                            f"{fraud_alert} | "
                            f"Text="
                            f"{transcript}"
                        )

                except Exception as exc:

                    print(
                        f"[FRAUD ERROR] "
                        f"{participant_identity}: "
                        f"{exc}"
                    )

                # =================================================
                # ECAPA SPEAKER VERIFICATION
                # =================================================

                speaker_similarity = None

                speaker_risk = None

                speaker_status = (
                    "NOT_AVAILABLE"
                )

                speaker_error = None

                if speaker_reference_available:

                    try:

                        speaker_result = (
                            verify_live_chunk(

                                REGISTERED_VOICE_PATH,

                                chunk,

                                sample_rate=TARGET_SR,
                            )
                        )

                        speaker_similarity = (
                            float(
                                speaker_result[
                                    "similarity"
                                ]
                            )
                        )

                        speaker_risk = round(
                            100.0
                            - speaker_similarity,
                            2,
                        )

                        speaker_status = (
                            speaker_result[
                                "status"
                            ]
                        )

                        print(
                            f"[ECAPA] "
                            f"{participant_identity} | "
                            f"Similarity="
                            f"{speaker_similarity:.2f}% | "
                            f"Risk="
                            f"{speaker_risk:.2f} | "
                            f"Status="
                            f"{speaker_status}"
                        )

                    except Exception as exc:

                        speaker_error = str(
                            exc
                        )

                        print(
                            f"[ECAPA ERROR] "
                            f"{participant_identity}: "
                            f"{speaker_error}"
                        )

                # =================================================
                # FINAL RISK FUSION
                # =================================================

                if speaker_risk is None:

                    # If ECAPA unavailable:
                    # AASIST + Fraud

                    fused_risk = (
                        0.75 * acoustic_risk
                        + 0.25 * fraud_risk
                    )

                    fusion_mode = (
                        "AASIST_75_FRAUD_25"
                    )

                else:

                    fused_risk = (
                        fuse_all_risks(
                            acoustic_risk,
                            speaker_risk,
                            fraud_risk,
                        )
                    )

                    fusion_mode = (
                        "AASIST_60_ECAPA_20_FRAUD_20"
                    )

                # =================================================
                # CENTRAL RISK ENGINE
                # =================================================

                risk_result = (
                    calculate_voice_risk(
                        fused_risk
                    )
                )

                final_risk = float(
                    risk_result[
                        "risk_score"
                    ]
                )

                status = str(
                    risk_result[
                        "risk_level"
                    ]
                )

                recommendation = str(
                    risk_result[
                        "recommendation"
                    ]
                )

                # =================================================
                # PROTECTION
                # =================================================

                (
                    protection_action,
                    verification_required,
                ) = get_protection_action(
                    status
                )

                # =================================================
                # PROTECTION LATCH
                # =================================================
                #
                # Once CRITICAL is detected, preserve the complete
                # CRITICAL event so later LOW/MEDIUM/HIGH chunks
                # cannot overwrite the important event.
                # =================================================

                if status == "CRITICAL":

                    transaction_frozen = True
                    critical_latched = True

                    critical_snapshot = {
                        "risk_level": "CRITICAL",
                        "risk_score": float(final_risk),
                        "fused_risk": float(final_risk),

                        # AASIST
                        "spoof_probability": float(spoof * 100.0),
                        "genuine_probability": float(genuine * 100.0),
                        "acoustic_risk": float(acoustic_risk),

                        # ECAPA
                        "speaker_similarity": (
                            float(speaker_similarity)
                            if speaker_similarity is not None
                            else None
                        ),
                        "speaker_risk": (
                            float(speaker_risk)
                            if speaker_risk is not None
                            else None
                        ),
                        "speaker_status": speaker_status,

                        # Fraud intelligence
                        "fraud_risk": float(fraud_risk),
                        "fraud_alert": fraud_alert,
                        "transcript": transcript,
                        "fraud_patterns": list(fraud_patterns),
                        "fraud_keywords": list(fraud_keywords),

                        # Security
                        "protection_action": "TRANSACTION_FROZEN",
                        "verification_required": True,
                        "recommendation": (
                            "Critical fraud intent detected. "
                            "Block sensitive action and "
                            "perform strong secondary verification."
                        ),

                        "participant_identity": participant_identity,
                        "timestamp": time.time(),
                    }

                # =================================================
                # EXTRA FRAUD OVERRIDE
                # =================================================

                if fraud_risk >= 70:

                    if final_risk < 80:
                        final_risk = 80.0

                    status = "CRITICAL"

                    protection_action = "TRANSACTION_FROZEN"

                    verification_required = True

                    transaction_frozen = True

                    recommendation = (
                        "Critical fraud intent detected. "
                        "Block sensitive action and "
                        "perform strong secondary verification."
                    )

                    # Fraud detection can independently create
                    # a CRITICAL event, so save the FINAL state.
                    critical_latched = True

                    critical_snapshot = {
                        "risk_level": "CRITICAL",
                        "risk_score": float(final_risk),
                        "fused_risk": float(final_risk),

                        # AASIST
                        "spoof_probability": float(spoof * 100.0),
                        "genuine_probability": float(genuine * 100.0),
                        "acoustic_risk": float(acoustic_risk),

                        # ECAPA
                        "speaker_similarity": (
                            float(speaker_similarity)
                            if speaker_similarity is not None
                            else None
                        ),
                        "speaker_risk": (
                            float(speaker_risk)
                            if speaker_risk is not None
                            else None
                        ),
                        "speaker_status": speaker_status,

                        # Fraud intelligence
                        "fraud_risk": float(fraud_risk),
                        "fraud_alert": fraud_alert,
                        "transcript": transcript,
                        "fraud_patterns": list(fraud_patterns),
                        "fraud_keywords": list(fraud_keywords),

                        # Security
                        "protection_action": "TRANSACTION_FROZEN",
                        "verification_required": True,
                        "recommendation": recommendation,

                        "participant_identity": participant_identity,
                        "timestamp": time.time(),
                    }

                elif fraud_risk >= 45:

                    if final_risk < 60:
                        final_risk = 60.0

                    status = "HIGH"

                    protection_action = "VERIFICATION_REQUIRED"

                    verification_required = True

                # =================================================
                # LATCHED CRITICAL STATE
                # =================================================
                #
                # After CRITICAL, later LOW/MEDIUM/HIGH chunks
                # cannot replace the saved CRITICAL analysis.
                # =================================================

                if critical_latched and critical_snapshot is not None:

                    transaction_frozen = True

                    status = "CRITICAL"

                    final_risk = critical_snapshot["risk_score"]

                    fused_risk = critical_snapshot["fused_risk"]

                    acoustic_risk = (
                        critical_snapshot["acoustic_risk"]
                    )

                    speaker_risk = (
                        critical_snapshot["speaker_risk"]
                    )

                    fraud_risk = (
                        critical_snapshot["fraud_risk"]
                    )

                    fraud_alert = (
                        critical_snapshot["fraud_alert"]
                    )

                    transcript = (
                        critical_snapshot["transcript"]
                    )

                    fraud_patterns = (
                        critical_snapshot["fraud_patterns"]
                    )

                    fraud_keywords = (
                        critical_snapshot["fraud_keywords"]
                    )

                    protection_action = (
                        "TRANSACTION_FROZEN"
                    )

                    verification_required = True

                    recommendation = (
                        critical_snapshot["recommendation"]
                    )

                # =================================================
                # CONSOLE OUTPUT
                # =================================================

                print(
                    "\n"
                    + "-" * 70
                )

                print(
                    "[SWARASHIELD LIVE ANALYSIS]"
                )

                print(
                    f"Participant       : "
                    f"{participant_identity}"
                )

                print(
                    f"AASIST Spoof      : "
                    f"{spoof:.2%}"
                )

                print(
                    f"AASIST Genuine    : "
                    f"{genuine:.2%}"
                )

                print(
                    f"Acoustic Risk     : "
                    f"{acoustic_risk:.2f}%"
                )

                if speaker_similarity is not None:

                    print(
                        f"ECAPA Similarity  : "
                        f"{speaker_similarity:.2f}%"
                    )

                    print(
                        f"Speaker Risk      : "
                        f"{speaker_risk:.2f}%"
                    )

                else:

                    print(
                        "ECAPA Similarity  : N/A"
                    )

                print(
                    f"Fraud Risk        : "
                    f"{fraud_risk:.2f}%"
                )

                print(
                    f"Fraud Alert       : "
                    f"{fraud_alert}"
                )

                print(
                    f"Transcript        : "
                    f"{transcript}"
                )

                if fraud_patterns:

                    print(
                        f"Fraud Patterns    : "
                        f"{', '.join(fraud_patterns)}"
                    )

                print(
                    f"Fused Risk        : "
                    f"{final_risk:.2f}%"
                )

                print(
                    f"Security Status   : "
                    f"{status}"
                )

                print(
                    f"Protection Action : "
                    f"{protection_action}"
                )

                print(
                    f"Transaction Frozen: "
                    f"{transaction_frozen}"
                )

                print(
                    "-" * 70
                    + "\n"
                )

                # =================================================
                # DASHBOARD PAYLOAD
                # =================================================

                payload = {

                    "timestamp": time.time(),

                    "source": "livekit",

                    "participant": (
                        participant_identity
                    ),

                    "room": ROOM_NAME,

                    "sample_rate": TARGET_SR,

                    "chunk_seconds": (
                        CHUNK_SECONDS
                    ),

                    # -----------------------------------------
                    # AASIST
                    # -----------------------------------------

                    "spoof_probability": round(
                        spoof,
                        6,
                    ),

                    "genuine_probability": round(
                        genuine,
                        6,
                    ),

                    "raw_acoustic_risk": round(
                        raw_acoustic_risk,
                        2,
                    ),

                    "acoustic_risk": round(
                        acoustic_risk,
                        2,
                    ),

                    # -----------------------------------------
                    # ECAPA
                    # -----------------------------------------

                    "speaker_verification": (
                        speaker_similarity
                        is not None
                    ),

                    "speaker_similarity": (

                        round(
                            speaker_similarity,
                            2,
                        )

                        if speaker_similarity
                        is not None

                        else None
                    ),

                    "speaker_risk": (

                        round(
                            speaker_risk,
                            2,
                        )

                        if speaker_risk
                        is not None

                        else None
                    ),

                    "speaker_status": (
                        speaker_status
                    ),

                    "speaker_error": (
                        speaker_error
                    ),

                    # -----------------------------------------
                    # FRAUD INTELLIGENCE
                    # -----------------------------------------

                    "transcript": (
                        transcript
                    ),

                    "fraud_risk": round(
                        fraud_risk,
                        2,
                    ),

                    "fraud_alert": (
                        fraud_alert
                    ),

                    "fraud_patterns": (
                        fraud_patterns
                    ),

                    "fraud_keywords": (
                        fraud_keywords
                    ),

                    # -----------------------------------------
                    # FUSION
                    # -----------------------------------------

                    "fusion_mode": (
                        fusion_mode
                    ),

                    "risk_score": round(
                        final_risk,
                        2,
                    ),

                    "risk_level": (
                        status
                    ),

                    # Existing dashboard
                    # compatibility

                    "acoustic_risk": round(
                        final_risk,
                        2,
                    ),

                    "status": (
                        status
                    ),

                    "recommendation": (
                        recommendation
                    ),

                    # -----------------------------------------
                    # PROTECTION
                    # -----------------------------------------

                    "protection_action": (
                        protection_action
                    ),

                    "verification_required": (
                        verification_required
                    ),

                    "transaction_frozen": (
                        transaction_frozen
                    ),

                    # -----------------------------------------
                    # CRITICAL LATCH
                    # -----------------------------------------

                    "critical_latched": (
                        critical_latched
                    ),

                    "critical_snapshot": (
                        critical_snapshot
                    ),

                    # -----------------------------------------
                    # HISTORY
                    # -----------------------------------------

                    "risk_history": [

                        round(
                            x,
                            2,
                        )

                        for x in risk_history
                    ],
                }

                write_status(
                    payload
                )

    except asyncio.CancelledError:

        print(
            f"[AUDIO] Analysis cancelled for "
            f"{participant_identity}"
        )

        raise

    except Exception as exc:

        print(
            f"[AUDIO ERROR] "
            f"{participant_identity}: "
            f"{exc}"
        )

    finally:

        await audio_stream.aclose()

        print(
            f"[AUDIO] Analysis stopped for "
            f"{participant_identity}"
        )


# ============================================================
# MAIN
# ============================================================

async def main():

    print("=" * 70)

    print(
        " SwaraShield LiveKit Interceptor"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # CHECK AASIST
    # --------------------------------------------------------

    if not Path(
        MODEL_PATH
    ).exists():

        raise FileNotFoundError(
            "AASIST model not found: "
            f"{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # LIVEKIT URL
    # --------------------------------------------------------

    url = os.getenv(
        "LIVEKIT_URL",
        "",
    ).strip()

    if not url:

        raise RuntimeError(
            "LIVEKIT_URL is required."
        )

    # --------------------------------------------------------
    # LIVEKIT CREDENTIALS
    # --------------------------------------------------------

    key = os.getenv(
        "LIVEKIT_API_KEY",
        "",
    ).strip()

    secret = os.getenv(
        "LIVEKIT_API_SECRET",
        "",
    ).strip()

    if not key or not secret:

        raise RuntimeError(
            "LIVEKIT_API_KEY / "
            "LIVEKIT_API_SECRET are required."
        )

    print(
        f"Room: {ROOM_NAME}"
    )

    print(
        "LiveKit configuration loaded."
    )

    # --------------------------------------------------------
    # LOAD AASIST
    # --------------------------------------------------------

    print(
        "Loading AASIST model..."
    )

    session = ort.InferenceSession(

        MODEL_PATH,

        providers=[
            "CPUExecutionProvider"
        ],
    )

    print(
        "AASIST model loaded."
    )

    # --------------------------------------------------------
    # CREATE ROOM
    # --------------------------------------------------------

    room = rtc.Room()

    # --------------------------------------------------------
    # TRACK SUBSCRIBED
    # --------------------------------------------------------

    @room.on(
        "track_subscribed"
    )
    def on_track_subscribed(

        track,

        publication,

        participant,
    ):

        if (
            track.kind
            == rtc.TrackKind.KIND_AUDIO
        ):

            print(
                "[LIVEKIT] Audio subscribed: "
                f"{participant.identity} / "
                f"{publication.name}"
            )

            asyncio.create_task(

                process_track(

                    track,

                    participant.identity,

                    session,
                )
            )

    # --------------------------------------------------------
    # TOKEN
    # --------------------------------------------------------

    token = make_worker_token()

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    write_status({

        "timestamp": time.time(),

        "source": "livekit",

        "status": "CONNECTING",

        "room": ROOM_NAME,
    })

    print(
        "Connecting to LiveKit..."
    )

    # --------------------------------------------------------
    # CONNECT
    # --------------------------------------------------------

    await room.connect(

        url,

        token,
    )

    write_status({

        "timestamp": time.time(),

        "source": "livekit",

        "status": "CONNECTED",

        "room": ROOM_NAME,
    })

    print("=" * 70)

    print(
        " SwaraShield LiveKit Interceptor"
    )

    print("=" * 70)

    print(
        f"Room: {ROOM_NAME}"
    )

    print(
        "Connected successfully."
    )

    print(
        "Waiting for participant audio..."
    )

    print(
        "Live analysis:"
    )

    print(
        "  AASIST  → Voice spoof detection"
    )

    print(
        "  ECAPA   → Speaker verification"
    )

    print(
        "  Whisper → Live transcription"
    )

    print(
        "  Fraud   → Scam intent detection"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # KEEP WORKER ALIVE
    # --------------------------------------------------------

    try:

        while room.isconnected():

            await asyncio.sleep(1)

    finally:

        await room.disconnect()

        print(
            "Disconnected from LiveKit."
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )
