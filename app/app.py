import io
import re
import random
import os
import tempfile
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st

st.markdown('\n<style>\n:root{\n --ss-bg:#07111f;--ss-panel:rgba(12,25,43,.88);\n --ss-border:rgba(84,211,255,.22);--ss-cyan:#55d9ff;\n --ss-green:#47e6a1;--ss-red:#ff5f6d;--ss-text:#e8f4ff;--ss-muted:#8fa8bd\n}\nhtml,body,[data-testid="stAppViewContainer"]{\n background:radial-gradient(circle at 20% 0%,#102b47 0%,var(--ss-bg) 42%,#030812 100%)!important;\n color:var(--ss-text)\n}\n[data-testid="stHeader"]{background:transparent!important}\n.block-container{max-width:1500px!important;padding-top:1.2rem!important;padding-bottom:2rem!important}\n.ss-topbar{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 18px;margin-bottom:18px;border:1px solid var(--ss-border);border-radius:16px;background:linear-gradient(135deg,rgba(15,35,58,.94),rgba(5,15,27,.94));box-shadow:0 10px 35px rgba(0,0,0,.24)}\n.ss-brand{font-size:24px;font-weight:800;letter-spacing:.3px}\n.ss-live{color:var(--ss-green);font-weight:700}\n.ss-live:before{content:"●";margin-right:7px}\n.ss-badge{border:1px solid rgba(85,217,255,.25);border-radius:999px;padding:6px 11px;color:var(--ss-cyan);font-size:13px}\n.ss-card{border:1px solid var(--ss-border);border-radius:16px;padding:18px;margin-bottom:16px;background:var(--ss-panel);box-shadow:0 10px 30px rgba(0,0,0,.18)}\n.ss-verdict{border-radius:18px;padding:24px;margin-bottom:16px;border:1px solid rgba(255,95,109,.28);background:linear-gradient(135deg,rgba(90,20,35,.72),rgba(22,15,30,.92))}\n.ss-score{font-size:42px;font-weight:900;line-height:1}\n.ss-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}\n.ss-pillar{padding:14px;border:1px solid var(--ss-border);border-radius:13px;background:rgba(7,18,31,.72)}\n.ss-pillar .name{font-size:12px;color:var(--ss-muted)}\n.ss-pillar .value{font-size:24px;font-weight:800;margin-top:4px}\n@media(max-width:900px){.ss-topbar{flex-wrap:wrap}.ss-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}\n@media(max-width:600px){.block-container{padding-left:.8rem!important;padding-right:.8rem!important}.ss-brand{font-size:20px}.ss-grid{grid-template-columns:1fr}.ss-score{font-size:34px}}\n</style>\n', unsafe_allow_html=True)

try:
    from speaker_verification import verify_speaker
    SPEAKER_VERIFICATION_AVAILABLE = True
except Exception:
    verify_speaker = None
    SPEAKER_VERIFICATION_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/aasist/aasist.onnx"
AUDIO_PATH = "data/sample_audio/test_pcm.wav"
TARGET_LENGTH = 64600


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SwaraShield",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def load_model():

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )

    return session


# ============================================================
# AUDIO LOADER
# ============================================================

def load_audio_from_bytes(file_bytes):

    audio, sample_rate = sf.read(
        io.BytesIO(file_bytes),
        dtype="float32",
    )

    # Stereo -> Mono
    if audio.ndim > 1:
        audio = np.mean(
            audio,
            axis=1,
        )

    # Normalize
    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    return audio, sample_rate


# ============================================================
# AUDIO PREPARATION
# ============================================================

def prepare_audio(audio):

    if len(audio) >= TARGET_LENGTH:

        audio = audio[:TARGET_LENGTH]

    else:

        padding = TARGET_LENGTH - len(audio)

        audio = np.pad(
            audio,
            (0, padding),
        )

    return audio.astype(np.float32)


def save_uploaded_audio_temp(file_bytes, prefix):

    temp = tempfile.NamedTemporaryFile(
        suffix=".wav",
        prefix=prefix,
        delete=False,
    )

    try:
        temp.write(file_bytes)
        return temp.name
    finally:
        temp.close()


# ============================================================
# AASIST VOICE DETECTION
# ============================================================

def detect_voice(
    session,
    audio,
):

    input_name = session.get_inputs()[0].name

    model_input = audio.reshape(
        1,
        TARGET_LENGTH,
    )

    outputs = session.run(
        None,
        {
            input_name: model_input,
        },
    )

    logits = outputs[0]

    # Softmax
    exp_logits = np.exp(
        logits - np.max(
            logits,
            axis=1,
            keepdims=True,
        )
    )

    probabilities = (
        exp_logits
        / np.sum(
            exp_logits,
            axis=1,
            keepdims=True,
        )
    )

    return logits, probabilities


# ============================================================
# FRAUD PHRASE INTELLIGENCE
# ============================================================

def analyze_fraud_context(text):

    text = text.lower().strip()

    if not text:
        return {
            "risk_score": 0.0,
            "signals": [],
            "matched_phrases": [],
        }

    rules = [
        (
            "Urgency / pressure language",
            [
                "jaldi",
                "immediately",
                "urgent",
                "urgently",
                "right now",
                "abhi",
                "turant",
            ],
            20,
        ),
        (
            "OTP or security credential request",
            [
                "otp",
                "one time password",
                "password",
                "pin",
                "cvv",
                "verification code",
                "share code",
            ],
            35,
        ),
        (
            "Money transfer request",
            [
                "transfer",
                "send money",
                "paise bhejo",
                "paise transfer",
                "fund transfer",
                "payment karo",
                "payment approve",
            ],
            20,
        ),
        (
            "Secrecy instruction",
            [
                "kisi ko mat batana",
                "don't tell anyone",
                "do not tell anyone",
                "secret",
                "confidential",
                "chup raho",
            ],
            25,
        ),
        (
            "Authority impersonation cue",
            [
                "bank se bol raha",
                "bank se bol rahi",
                "i am calling from the bank",
                "police se bol raha",
                "government se",
                "official speaking",
            ],
            15,
        ),
    ]

    signals = []
    matched_phrases = []
    risk_score = 0

    for signal_name, phrases, weight in rules:

        matched = []

        for phrase in phrases:

            if phrase in text:
                matched.append(phrase)

        if matched:

            signals.append(signal_name)
            matched_phrases.extend(matched)
            risk_score += weight

    # Remove duplicates
    matched_phrases = list(
        dict.fromkeys(matched_phrases)
    )

    return {
        "risk_score": float(
            min(risk_score, 100)
        ),
        "signals": signals,
        "matched_phrases": matched_phrases,
    }


# ============================================================
# TRANSACTION CONTEXT RISK
# ============================================================

def get_transaction_risk(
    transaction_type,
    transaction_amount,
    new_beneficiary,
    unusual_time,
    location_change,
    multiple_recent_transactions,
):

    # Action-based risk
    if transaction_type == "No sensitive action":
        action_risk = 0

    elif transaction_type == "Information request":
        action_risk = 10

    elif transaction_type == "Account change":
        action_risk = 30

    elif transaction_type == "Payment approval":
        action_risk = 45

    elif transaction_type == "Fund transfer":
        action_risk = 55

    elif transaction_type == "Privileged access approval":
        action_risk = 60

    else:
        action_risk = 0

    # Amount-based risk
    amount_risk = 0

    if transaction_amount >= 10000000:
        amount_risk = 35

    elif transaction_amount >= 1000000:
        amount_risk = 25

    elif transaction_amount >= 100000:
        amount_risk = 15

    elif transaction_amount >= 10000:
        amount_risk = 5

    # Additional signals
    beneficiary_risk = (
        20 if new_beneficiary else 0
    )

    time_risk = (
        10 if unusual_time else 0
    )

    location_risk = (
        15 if location_change else 0
    )

    velocity_risk = (
        15 if multiple_recent_transactions else 0
    )

    total_context_risk = min(
        action_risk
        + amount_risk
        + beneficiary_risk
        + time_risk
        + location_risk
        + velocity_risk,
        100,
    )

    return float(total_context_risk)


# ============================================================
# CONTEXT + FRAUD PHRASE FUSION
# ============================================================

def combine_context_risk(
    transaction_risk,
    fraud_phrase_risk,
):

    combined_risk = (
        0.70 * transaction_risk
        + 0.30 * fraud_phrase_risk
    )

    return float(
        min(combined_risk, 100)
    )


# ============================================================
# FINAL RISK ENGINE
# ============================================================
def get_livekit_risk():
    status_path = Path("data/live/live_status.json")

    try:
        if not status_path.exists():
            return None

        with open(status_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return None
def calculate_final_risk(
    spoof_probability,
    context_risk,
):

    voice_risk = spoof_probability * 100
        # --------------------------------------------------------
    # LIVEKIT OVERRIDE
    # --------------------------------------------------------
    livekit_data = get_livekit_risk()

    if livekit_data is not None:
        spoof_probability = float(
            livekit_data.get(
                "spoof_probability",
                spoof_probability
            )
        )

    voice_risk = spoof_probability * 100

    # Voice = 70%
    # Context + language = 30%
    final_score = (
        voice_risk * 0.70
        + context_risk * 0.30
    )

    final_score = min(
        max(final_score, 0),
        100,
    )

    if final_score >= 80:

        risk_level = "CRITICAL"

        recommendation = (
            "BLOCK sensitive action and "
            "perform independent verification."
        )

    elif final_score >= 60:

        risk_level = "HIGH"

        recommendation = (
            "STOP sensitive action and "
            "perform secondary verification."
        )

    elif final_score >= 35:

        risk_level = "MEDIUM"

        recommendation = (
            "Proceed with caution and "
            "perform additional verification."
        )

    else:

        risk_level = "LOW"

        recommendation = (
            "Voice appears lower-risk. "
            "Continue normal verification."
        )

    return (
        voice_risk,
        final_score,
        risk_level,
        recommendation,
    )
# ============================================================
# TRANSACTION PROTECTION — SIH DEMO
# ============================================================
def get_protection_action(risk_level):
    if risk_level == "CRITICAL":
        return {
            "action": "TRANSACTION_FROZEN",
            "message": (
                "🚨 Critical voice-fraud risk detected. "
                "Sensitive transaction has been frozen."
            ),
            "verification_required": True,
            "verification_status": "REQUIRED",
            "verification_method": "INDEPENDENT_VERIFICATION",
        }

    if risk_level == "HIGH":
        return {
            "action": "VERIFICATION_REQUIRED",
            "message": (
                "🔴 High-risk call detected. "
                "Stop sensitive action and verify independently."
            ),
            "verification_required": True,
            "verification_status": "REQUIRED",
            "verification_method": "INDEPENDENT_VERIFICATION",
        }

    if risk_level == "MEDIUM":
        return {
            "action": "WARNING",
            "message": (
                "🟠 Suspicious activity detected. "
                "Additional verification recommended."
            ),
            "verification_required": True,
            "verification_status": "RECOMMENDED",
            "verification_method": "SECONDARY_VERIFICATION",
        }

    return {
        "action": "ALLOW",
        "message": (
            "🟢 No critical voice-fraud risk detected. "
            "Normal verification procedures may continue."
        ),
        "verification_required": False,
        "verification_status": "NOT_REQUIRED",
        "verification_method": None,
    }
# ============================================================
# SENTINEL AI COPILOT
# ============================================================

def get_copilot_reply(question):

    q = question.lower().strip()

    if any(word in q for word in [
        "kaise kaam", "how does", "working", "work",
        "workflow", "flow", "app kya", "kya karta"
    ]):
        return (
            "SwwaraShield ka workflow hai: Voice Input → "
            "AASIST + ONNX voice authenticity analysis → "
            "transaction/context risk → fraud phrase intelligence → "
            "voice + context risk fusion → LOW/MEDIUM/HIGH/CRITICAL "
            "decision → high-risk par Adaptive Challenge Verification."
        )

    if any(word in q for word in [
        "aasist", "onnx", "model", "ai engine", "voice detection"
    ]):
        return (
            "AI Engine AASIST anti-spoofing model aur ONNX Runtime use karta hai. "
            "Uploaded audio ko fixed model input length ke liye prepare karke "
            "inference chalaya jata hai aur spoof/bonafide probabilities nikali "
            "jati hain."
        )

    if any(word in q for word in [
        "risk", "score", "high", "critical", "medium", "low"
    ]):
        return (
            "Risk Engine voice risk aur context risk ko fuse karta hai. "
            "LOW result normal verification ke saath continue hota hai, MEDIUM par "
            "additional verification, HIGH par STOP & VERIFY, aur CRITICAL par "
            "BLOCK ACTION recommend kiya jata hai."
        )

    if any(word in q for word in [
        "context", "transaction", "beneficiary", "amount",
        "location", "time", "multiple"
    ]):
        return (
            "Context Risk requested action, transaction amount, new/unknown "
            "beneficiary, unusual transaction time, unusual location/device change, "
            "aur multiple recent transactions jaise signals se calculate hota hai."
        )

    if any(word in q for word in [
        "phrase", "language", "otp", "urgent", "jaldi", "fraud"
    ]):
        return (
            "Fraud Phrase Intelligence entered Hindi, English aur Hinglish text mein "
            "urgency, OTP/security credential requests, money transfer requests aur "
            "confidential-information patterns ko check karta hai. Matched signals "
            "context risk mein contribute karte hain."
        )

    if any(word in q for word in [
        "challenge", "verification phrase", "adaptive"
    ]):
        return (
            "Adaptive Challenge Verification sirf HIGH ya CRITICAL risk par activate "
            "hota hai. App ek random phrase generate karta hai aur prototype mein "
            "caller se us phrase ko repeat karne ke liye kaha jata hai. Future version "
            "mein repeated response par speaker consistency aur liveness checks add "
            "kiye ja sakte hain."
        )

    if any(word in q for word in [
        "privacy", "audio store", "retention", "data"
    ]):
        return (
            "Current prototype local/edge inference aur minimal audio retention approach "
            "dikhata hai. Is demo workflow mein uploaded audio ko analysis ke liye process "
            "kiya jata hai; production deployment mein retention, consent aur access "
            "controls explicitly enforce karne honge."
        )

    if any(word in q for word in [
        "result", "latest", "current", "abhi", "final decision"
    ]):
        latest = st.session_state.get("latest_result")

        if latest:
            return (
                f"Latest available analysis: Voice Risk {latest['voice_risk']:.2f}%, "
                f"Context Risk {latest['context_risk']:.2f}%, "
                f"Final Risk {latest['final_score']:.2f}, "
                f"Level {latest['risk_level']}."
            )

        return (
            "Abhi koi analysis result available nahi hai. Pehle voice sample upload "
            "karke Analyze Voice & Transaction run karo."
        )

    return (
        "Main SwaraShield AI Copilot hoon. Main is app ka workflow, AASIST + ONNX, "
        "risk score, transaction/context signals, fraud phrase intelligence, privacy, "
        "Adaptive Challenge Verification aur current result explain kar sakta hoon. "
        "Aap Hindi, Hinglish ya English mein pooch sakte ho."
    )

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🛡️ SwaraShield")

    st.write(
        "AI-powered voice integrity and "
        "impersonation defense."
    )

    st.divider()

    st.write("**AI Engine**")
    st.write("AASIST + ONNX Runtime")

    st.write("**Risk Engine**")
    st.write("Voice + Context + Language Fusion")

    st.write("**Inference**")
    st.write("Local / Edge")

    st.write("**Privacy**")
    st.write("Minimal audio retention")

    st.divider()

    st.caption(
        "SwaraShield"
    )


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ SwaraShield")

st.subheader(
    "AI-Powered Real-Time Voice Cloning & "
    "Impersonation Defense"
)

st.write(
    "AI-powered voice integrity verification with "
    "context-aware fraud prevention."
)

st.divider()


# ============================================================
# TRANSACTION CONTEXT
# ============================================================

st.subheader("🏦 Transaction Context")

context_col1, context_col2 = st.columns(2)

with context_col1:

    transaction_type = st.selectbox(
        "Action requested by caller",
        [
            "No sensitive action",
            "Information request",
            "Account change",
            "Payment approval",
            "Fund transfer",
            "Privileged access approval",
        ],
    )


with context_col2:

    transaction_amount = st.number_input(
        "Transaction amount (₹)",
        min_value=0,
        value=0,
        step=1000,
    )


st.caption(
    "Additional context signals help simulate "
    "real-world fraud and impersonation scenarios."
)


signal_col1, signal_col2 = st.columns(2)

with signal_col1:

    new_beneficiary = st.checkbox(
        "👤 New / unknown beneficiary",
        value=False,
    )

    unusual_time = st.checkbox(
        "🌙 Unusual transaction time",
        value=False,
    )


with signal_col2:

    location_change = st.checkbox(
        "📍 Unusual location / device change",
        value=False,
    )

    multiple_recent_transactions = st.checkbox(
        "🔁 Multiple recent transactions",
        value=False,
    )


st.divider()


# ============================================================
# VOICE INPUT
# ============================================================

st.subheader("🎙 Voice Input")

uploaded_file = st.file_uploader(
    "Upload voice sample",
    type=["wav"],
    help=(
        "Upload a 16 kHz WAV file for "
        "voice authenticity analysis."
    ),
)


# ============================================================
# SPEAKER CONSISTENCY CHECK
# ============================================================

st.markdown("### 👤 Speaker Consistency Check")

registered_voice_file = st.file_uploader(
    "Upload registered genuine voice sample",
    type=["wav"],
    key="registered_voice_sample",
    help=(
        "Optional trusted voice sample used to compare "
        "the current caller voice."
    ),
)


# ============================================================
# CALLER TRANSCRIPT
# ============================================================

st.markdown("### 💬 Caller Message / Transcript")

caller_text = st.text_area(
    "Enter caller's spoken message or transcript",
    placeholder=(
        "Example: Main bank se bol raha hoon. "
        "Jaldi paise transfer karo aur kisi ko mat batana."
    ),
    height=120,
)

fraud_context = analyze_fraud_context(
    caller_text
)

fraud_phrase_risk = (
    fraud_context["risk_score"]
)

fraud_signals = (
    fraud_context["signals"]
)

matched_phrases = (
    fraud_context["matched_phrases"]
)


# ============================================================
# LIVE FRAUD CONTEXT PREVIEW
# ============================================================

if caller_text and caller_text.strip():

    st.markdown(
        "### 🧠 Fraud Context Intelligence"
    )

    fraud_col1, fraud_col2 = st.columns(2)

    with fraud_col1:

        st.metric(
            "Fraud Phrase Risk",
            f"{fraud_phrase_risk:.0f}%",
        )

    with fraud_col2:

        st.metric(
            "Detected Signals",
            len(fraud_signals),
        )

    if fraud_signals:

        st.markdown(
            "#### ⚠️ Detected Risk Signals"
        )

        for signal in fraud_signals:
            st.warning(f"⚠️ {signal}")

    else:

        st.success(
            "No high-risk fraud phrases detected."
        )

    if matched_phrases:

        st.markdown(
            "#### 🔎 Matched Phrases"
        )

        for phrase in matched_phrases:
            st.write(f"• `{phrase}`")


st.divider()


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 ANALYZE VOICE & TRANSACTION",
    use_container_width=True,
):

    try:

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        with st.spinner(
            "Loading AI detection model..."
        ):

            session = load_model()


        # ----------------------------------------------------
        # LOAD AUDIO
        # ----------------------------------------------------

        with st.spinner(
            "Processing voice sample..."
        ):

            if uploaded_file is not None:

                file_bytes = (
                    uploaded_file.getvalue()
                )

                audio, sample_rate = (
                    load_audio_from_bytes(
                        file_bytes
                    )
                )

                audio_source = (
                    uploaded_file.name
                )

            else:

                with open(
                    AUDIO_PATH,
                    "rb",
                ) as audio_file:

                    file_bytes = (
                        audio_file.read()
                    )

                audio, sample_rate = (
                    load_audio_from_bytes(
                        file_bytes
                    )
                )

                audio_source = (
                    "Built-in test audio"
                )


        # ----------------------------------------------------
        # SAMPLE RATE WARNING
        # ----------------------------------------------------

        if sample_rate != 16000:

            st.warning(
                f"Audio sample rate is {sample_rate} Hz. "
                "The AASIST prototype expects 16 kHz audio."
            )


        # ----------------------------------------------------
        # PREPARE AUDIO
        # ----------------------------------------------------

        prepared_audio = prepare_audio(
            audio
        )


        # ----------------------------------------------------
        # AASIST DETECTION
        # ----------------------------------------------------

        with st.spinner(
            "Running AI voice authenticity analysis..."
        ):

            logits, probabilities = detect_voice(
                session,
                prepared_audio,
            )


        # ----------------------------------------------------
        # CLASS MAPPING
        # ----------------------------------------------------
        #
        # Class 0 -> Spoof
        # Class 1 -> Genuine
        #
        # Based on the prototype test mapping.
        # ----------------------------------------------------

        spoof_probability = float(
            probabilities[0][0]
        )

        genuine_probability = float(
            probabilities[0][1]
        )


        # ----------------------------------------------------
        # SPEAKER CONSISTENCY VERIFICATION
        # ----------------------------------------------------

        speaker_result = None
        speaker_error = None
        temp_paths = []

        if registered_voice_file is not None:

            if not SPEAKER_VERIFICATION_AVAILABLE:

                speaker_error = (
                    "speaker_verification.py could not be loaded."
                )

            else:

                try:

                    with st.spinner(
                        "Checking speaker consistency..."
                    ):

                        registered_path = save_uploaded_audio_temp(
                            registered_voice_file.getvalue(),
                            "SwaraShield_registered_",
                        )

                        current_path = save_uploaded_audio_temp(
                            file_bytes,
                            "SwaraShield_current_",
                        )

                        temp_paths = [
                            registered_path,
                            current_path,
                        ]

                        speaker_result = verify_speaker(
                            registered_path,
                            current_path,
                        )

                except Exception as error:

                    speaker_error = str(error)

                finally:

                    for temp_path in temp_paths:

                        try:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                        except OSError:
                            pass


        # ----------------------------------------------------
        # TRANSACTION RISK
        # ----------------------------------------------------

        transaction_context_risk = (
            get_transaction_risk(
                transaction_type,
                transaction_amount,
                new_beneficiary,
                unusual_time,
                location_change,
                multiple_recent_transactions,
            )
        )


        # ----------------------------------------------------
        # FUSED CONTEXT RISK
        # ----------------------------------------------------

        context_risk = combine_context_risk(
            transaction_context_risk,
            fraud_phrase_risk,
        )


        # ----------------------------------------------------
        # FINAL RISK
        # ----------------------------------------------------

        (
            voice_risk,
            final_score,
            risk_level,
            recommendation,
        ) = calculate_final_risk(
            spoof_probability,
            context_risk,
        )

        st.session_state["latest_result"] = {
            "voice_risk": voice_risk,
            "context_risk": context_risk,
            "final_score": final_score,
            "risk_level": risk_level,
            "speaker_similarity": (
                float(speaker_result.get("similarity", 0.0))
                if speaker_result is not None
                else None
            ),
            "speaker_status": (
                speaker_result.get("status", "UNKNOWN")
                if speaker_result is not None
                else None
            ),
        }


        # ====================================================
        # RESULTS
        # ====================================================

        st.success(
            "Voice + transaction + fraud context "
            "analysis completed."
        )

        st.caption(
            f"Analyzed audio: {audio_source}"
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:

            st.metric(
                "🎙 Voice Risk",
                f"{voice_risk:.2f}%",
            )


        with col2:

            st.metric(
                "🏦 Context Risk",
                f"{context_risk:.2f}%",
            )


        with col3:

            st.metric(
                "🛡️ Final Risk",
                f"{final_score:.2f}",
            )


        with col4:

            st.metric(
                "💰 Amount",
                f"₹{transaction_amount:,.0f}",
            )


        with col5:

            if speaker_result is not None:
                speaker_match_text = (
                    f"{float(speaker_result.get('similarity', 0.0)):.2f}%"
                )
            else:
                speaker_match_text = "Not checked"

            st.metric(
                "👤 Speaker Match",
                speaker_match_text,
            )


        st.divider()


        # ----------------------------------------------------
        # SPEAKER CONSISTENCY RESULT
        # ----------------------------------------------------

        if speaker_result is not None:

            st.subheader(
                "👤 Speaker Consistency Result"
            )

            speaker_col1, speaker_col2, speaker_col3 = st.columns(3)

            similarity = float(
                speaker_result.get("similarity", 0.0)
            )

            speaker_status = speaker_result.get(
                "status",
                "UNKNOWN",
            )

            speaker_risk = speaker_result.get(
                "risk",
                "UNKNOWN",
            )

            with speaker_col1:
                st.metric(
                    "Similarity Score",
                    f"{similarity:.2f}%",
                )

            with speaker_col2:
                st.metric(
                    "Speaker Status",
                    speaker_status,
                )

            with speaker_col3:
                st.metric(
                    "Speaker Risk",
                    speaker_risk,
                )

            if str(speaker_risk).upper() == "HIGH":
                st.error(
                    "⚠️ Speaker mismatch detected. "
                    "Use independent verification."
                )
            elif str(speaker_risk).upper() == "MEDIUM":
                st.warning(
                    "⚠️ Speaker match needs additional verification."
                )
            else:
                st.success(
                    "✓ Current voice is consistent with the "
                    "registered genuine voice."
                )

        elif speaker_error:

            st.warning(
                "Speaker Consistency Check could not be completed: "
                + speaker_error
            )

        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        if risk_level == "CRITICAL":

            st.error(
                "🚨 CRITICAL RISK"
            )

        elif risk_level == "HIGH":

            st.error(
                "🔴 HIGH RISK"
            )

        elif risk_level == "MEDIUM":

            st.warning(
                "🟠 MEDIUM RISK"
            )

        else:

            st.success(
                "🟢 LOW RISK"
            )


        # ----------------------------------------------------
        # FINAL RISK GAUGE
        # ----------------------------------------------------

        st.subheader(
            "🛡️ Final Impersonation Risk"
        )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=final_score,
                title={
                    "text": "Risk Score"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "bar": {
                        "thickness": 0.75
                    },
                    "steps": [
                        {
                            "range": [0, 35]
                        },
                        {
                            "range": [35, 60]
                        },
                        {
                            "range": [60, 80]
                        },
                        {
                            "range": [80, 100]
                        },
                    ],
                },
            )
        )

        fig.update_layout(
            height=350,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


        # ----------------------------------------------------
        # RISK EXPLANATION
        # ----------------------------------------------------

        st.subheader(
            "🧠 Why did SwaraShield raise this risk?"
        )

        explanation_col1, explanation_col2 = (
            st.columns(2)
        )


        with explanation_col1:

            st.write(
                "### 🎙 Voice Analysis"
            )

            st.write(
                f"Spoof likelihood: "
                f"**{voice_risk:.2f}%**"
            )

            if voice_risk >= 50:

                st.write(
                    "The AI anti-spoofing model detected "
                    "characteristics associated with "
                    "synthetic or manipulated speech."
                )

            else:

                st.write(
                    "The analyzed voice is currently more "
                    "consistent with the bonafide class "
                    "under this prototype's model mapping."
                )

            st.write(
                f"Bonafide probability: "
                f"**{genuine_probability * 100:.2f}%**"
            )


        with explanation_col2:

            st.write(
                "### 🏦 Context Analysis"
            )

            st.write(
                f"Requested action: "
                f"**{transaction_type}**"
            )

            st.write(
                f"Transaction context risk: "
                f"**{transaction_context_risk:.2f}%**"
            )

            st.write(
                f"Fraud phrase risk: "
                f"**{fraud_phrase_risk:.2f}%**"
            )

            st.write(
                f"Fused context risk: "
                f"**{context_risk:.2f}%**"
            )

            if transaction_amount > 0:

                st.write(
                    f"Transaction amount: "
                    f"**₹{transaction_amount:,.0f}**"
                )

            active_signals = []

            if new_beneficiary:
                active_signals.append(
                    "New / unknown beneficiary"
                )

            if unusual_time:
                active_signals.append(
                    "Unusual transaction time"
                )

            if location_change:
                active_signals.append(
                    "Unusual location / device change"
                )

            if multiple_recent_transactions:
                active_signals.append(
                    "Multiple recent transactions"
                )

            if active_signals:

                st.write(
                    "Additional risk signals: "
                    f"**{', '.join(active_signals)}**"
                )

            else:

                st.write(
                    "Additional risk signals: "
                    "**None selected**"
                )


        # ----------------------------------------------------
        # FRAUD LANGUAGE EXPLANATION
        # ----------------------------------------------------

        if caller_text and caller_text.strip():

            st.divider()

            st.subheader(
                "🧠 Fraud Language Analysis"
            )

            if fraud_signals:

                for signal in fraud_signals:

                    st.warning(
                        f"⚠️ {signal}"
                    )

            else:

                st.success(
                    "No high-risk fraud language "
                    "signals were detected."
                )

            if matched_phrases:

                st.write(
                    "**Matched phrases:**"
                )

                st.write(
                    ", ".join(
                        matched_phrases
                    )
                )


        # ----------------------------------------------------
        # ADAPTIVE CHALLENGE VERIFICATION
        # ----------------------------------------------------

        if risk_level in ["HIGH", "CRITICAL"]:

            st.divider()

            st.subheader(
                "🎯 Adaptive Challenge Verification"
            )

            challenge_phrases = [
                "Blue mango seven",
                "River moon twenty",
                "Green tiger forty",
                "Silver cloud nine",
                "Orange lamp sixty",
                "Purple star twelve",
            ]

            challenge_phrase = random.choice(
                challenge_phrases
            )

            st.warning(
                "⚠️ High-risk voice detected. "
                "Ask the caller to repeat the verification phrase."
            )

            st.info(
                f"🔐 Verification Phrase: **{challenge_phrase}**"
            )

            st.caption(
                "Prototype security challenge. "
                "A production system would compare the repeated "
                "response using speaker consistency and liveness checks."
            )
        # ----------------------------------------------------
        # SECURITY DECISION
        # ----------------------------------------------------
    
        st.divider()

        st.subheader(
            "🔐 Security Decision"
        )

        if risk_level == "CRITICAL":

            st.error(
                "🚨 BLOCK ACTION"
            )

            st.write(
                "The requested sensitive action should not "
                "be completed until independent verification "
                "succeeds."
            )

            st.write(
                "Recommended:"
            )

            st.write(
                "• Independent callback"
            )

            st.write(
                "• Multi-factor authentication"
            )

            st.write(
                "• Supervisor / security escalation"
            )


        elif risk_level == "HIGH":

            st.warning(
                "⚠️ STOP & VERIFY"
            )

            st.write(
                "Do not approve the requested sensitive "
                "action without secondary verification."
            )

            st.write(
                "Recommended:"
            )

            st.write(
                "• Call back using a trusted number"
            )

            st.write(
                "• Confirm using MFA"
            )


        elif risk_level == "MEDIUM":

            st.warning(
                "🟠 CAUTION"
            )

            st.write(
                "Perform additional verification before "
                "continuing with the requested action."
            )


        else:

            st.success(
                "🟢 ALLOW WITH NORMAL VERIFICATION"
            )

            st.write(
                "The current prototype assessment is low-risk, "
                "but standard verification should still apply."
            )


        # ----------------------------------------------------
        # TECHNICAL DETAILS
        # ----------------------------------------------------

        st.divider()

        with st.expander(
            "🔬 Technical Analysis"
        ):

            st.write(
                f"Raw model logits: {logits}"
            )

            st.write(
                f"Spoof probability: "
                f"{spoof_probability * 100:.2f}%"
            )

            st.write(
                f"Bonafide probability: "
                f"{genuine_probability * 100:.2f}%"
            )

            st.write(
                f"Transaction context risk: "
                f"{transaction_context_risk:.2f}"
            )

            st.write(
                f"Fraud phrase risk: "
                f"{fraud_phrase_risk:.2f}"
            )

            st.write(
                f"Fused context risk: "
                f"{context_risk:.2f}"
            )

            st.write(
                f"Final risk score: "
                f"{final_score:.2f}"
            )


    except Exception as error:

        st.error(
            "Analysis failed."
        )

        st.exception(error)

# ============================================================
# SENTINEL AI COPILOT UI
# ============================================================

st.divider()

st.subheader("🤖 Sentinel AI Copilot")

st.caption(
    "Ask how SwaraShield works, what each risk means, "
    "how AASIST detection works, or why a verification step was triggered."
)

if "copilot_messages" not in st.session_state:
    st.session_state["copilot_messages"] = [
        {
            "role": "assistant",
            "content": (
                "Namaste! Main Sentinel AI Copilot hoon. "
                "Main SwaraShield app ka workflow aur features explain kar sakta hoon. "
                "Try: 'Ye app kaise kaam karta hai?'"
            ),
        }
    ]

for message in st.session_state["copilot_messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

copilot_question = st.chat_input(
    "Ask Sentinel AI Copilot about this app..."
)

if copilot_question:
    st.session_state["copilot_messages"].append(
        {
            "role": "user",
            "content": copilot_question,
        }
    )

    with st.chat_message("user"):
        st.write(copilot_question)

    copilot_reply = get_copilot_reply(
        
        copilot_question
    )

    st.session_state["copilot_messages"].append(
        {
            "role": "assistant",
            "content": copilot_reply,
        }
    )

    with st.chat_message("assistant"):
        st.write(copilot_reply)

