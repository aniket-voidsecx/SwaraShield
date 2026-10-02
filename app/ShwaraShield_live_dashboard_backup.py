import io
import re
import random
import os
import tempfile
import json

import numpy as np
import onnxruntime as ort
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st


# ============================================================
# OPTIONAL INTEGRATIONS
# ============================================================

try:
    from livekit_integration import render_livekit_panel
    LIVEKIT_UI_AVAILABLE = True
    LIVEKIT_UI_ERROR = ""
except Exception as e:
    render_livekit_panel = None
    LIVEKIT_UI_AVAILABLE = False
    LIVEKIT_UI_ERROR = str(e)


try:
    from speaker_verification import verify_speaker
    SPEAKER_VERIFICATION_AVAILABLE = True
except Exception:
    verify_speaker = None
    SPEAKER_VERIFICATION_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ShwaraShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS
# ============================================================

APP_CSS = """
<style>

:root {
    --bg: #07110c;
    --bg2: #0a1710;
    --panel: #0d1b13;
    --panel2: #102218;
    --line: rgba(102, 255, 145, 0.16);
    --line2: rgba(102, 255, 145, 0.28);
    --green: #63ff8b;
    --green2: #1ee66b;
    --text: #f1f7f2;
    --muted: #8b9b90;
    --blue: #5cc8ff;
    --red: #ff5f69;
    --orange: #ffb84d;
}

html,
body,
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(
            circle at 78% 0%,
            rgba(45, 145, 72, .12),
            transparent 32%
        ),
        radial-gradient(
            circle at 0% 40%,
            rgba(25, 92, 49, .10),
            transparent 35%
        ),
        var(--bg) !important;

    color: var(--text) !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
}

[data-testid="stToolbar"] {
    visibility: visible;
}

.block-container {
    max-width: 1380px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 4rem !important;
}

[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    display: none !important;
}

h1,
h2,
h3,
h4 {
    color: var(--text) !important;
    letter-spacing: -0.025em !important;
}

h1 {
    font-size: clamp(2.2rem, 5vw, 4rem) !important;
    font-weight: 900 !important;
    margin-bottom: .15rem !important;
}

h2 {
    font-size: 1.8rem !important;
    font-weight: 850 !important;
}

h3 {
    font-size: 1.25rem !important;
    font-weight: 800 !important;
}

p,
label,
.stMarkdown,
.stCaption {
    color: #c6d1c9 !important;
}

.hero {
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 28px 32px;
    margin: 0 0 24px 0;
    background:
        radial-gradient(
            circle at 92% 50%,
            rgba(81, 255, 122, .13),
            transparent 28%
        ),
        linear-gradient(
            135deg,
            rgba(17, 40, 25, .94),
            rgba(7, 18, 11, .96)
        );

    box-shadow: 0 18px 50px rgba(0,0,0,.28);
}

.hero-title {
    font-size: clamp(2.4rem, 6vw, 4.4rem);
    line-height: 1;
    font-weight: 950;
    color: #f4fff6;
}

.hero-title span {
    color: var(--green);
}

.hero-sub {
    margin-top: 12px;
    font-size: 1.08rem;
    color: #9eafa4;
}

.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;

    border: 1px solid rgba(99,255,139,.28);
    border-radius: 999px;

    padding: 8px 13px;

    color: var(--green);
    background: rgba(26, 91, 44, .16);

    font-weight: 800;
    font-size: .82rem;

    letter-spacing: .05em;
    text-transform: uppercase;
}

.live-dot {
    width: 9px;
    height: 9px;

    border-radius: 50%;

    background: var(--green);

    box-shadow:
        0 0 14px rgba(99,255,139,.9);
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background:
        linear-gradient(
            145deg,
            rgba(15, 31, 20, .98),
            rgba(8, 19, 12, .98)
        ) !important;

    border: 1px solid var(--line2) !important;

    border-radius: 18px !important;

    box-shadow:
        0 12px 34px rgba(0,0,0,.20) !important;
}

.feature-title {
    font-size: 1.25rem;
    font-weight: 850;
    margin-bottom: 8px;
}

.feature-text {
    color: #aab8ae;
    line-height: 1.6;
    min-height: 72px;
}

.engine-ready {
    margin-top: 14px;

    border-radius: 10px;

    padding: 12px 14px;

    background:
        rgba(40, 105, 59, .32);

    border:
        1px solid rgba(99,255,139,.12);

    color:
        #c8ffd6;

    font-weight:
        700;
}

.section-kicker {
    color: var(--muted);

    font-size: .78rem;

    font-weight: 850;

    letter-spacing: .16em;

    text-transform: uppercase;

    margin: 24px 0 10px;
}

.stSelectbox > div > div,
.stNumberInput > div > div,
.stTextArea textarea,
.stTextInput input {
    background: #0a1710 !important;

    color: #edf8ef !important;

    border:
        1px solid rgba(102,255,145,.18) !important;

    border-radius: 10px !important;
}

.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: var(--green) !important;

    box-shadow:
        0 0 0 1px rgba(99,255,139,.18) !important;
}

[data-baseweb="select"] > div {
    background: #0a1710 !important;

    border-color:
        rgba(102,255,145,.18) !important;
}

[data-testid="stFileUploader"] {
    background: #0a1710 !important;

    border:
        1px dashed rgba(99,255,139,.25) !important;

    border-radius: 14px !important;

    padding: 6px !important;
}

.stButton > button {
    border:
        1px solid rgba(99,255,139,.35) !important;

    border-radius: 11px !important;

    background:
        linear-gradient(
            135deg,
            #12351e,
            #0b2314
        ) !important;

    color:
        #eaffee !important;

    font-weight:
        800 !important;

    min-height:
        44px !important;

    transition:
        all .18s ease !important;
}

.stButton > button:hover {
    border-color:
        var(--green) !important;

    background:
        linear-gradient(
            135deg,
            #195b2d,
            #0d3019
        ) !important;

    box-shadow:
        0 0 24px rgba(99,255,139,.18) !important;

    transform:
        translateY(-1px);
}

[data-testid="stMetric"] {
    background:
        rgba(13, 30, 18, .82);

    border:
        1px solid rgba(99,255,139,.14);

    border-radius:
        14px;

    padding:
        15px 16px;
}

[data-testid="stMetricLabel"] {
    color:
        #899c8f !important;
}

[data-testid="stMetricValue"] {
    color:
        #effff3 !important;

    font-weight:
        900 !important;
}

[data-testid="stAlert"] {
    border-radius:
        12px !important;

    background:
        rgba(14, 31, 19, .9) !important;
}

[data-testid="stCheckbox"] label p {
    color:
        #c7d3ca !important;
}

hr {
    border-color:
        rgba(99,255,139,.12) !important;
}

[data-testid="stPopover"] > div > div {
    background:
        #0b1710 !important;

    border:
        1px solid rgba(99,255,139,.25) !important;

    border-radius:
        16px !important;
}

.js-plotly-plot,
.plot-container {
    border-radius:
        16px !important;
}

@media (max-width: 900px) {

    .block-container {
        padding-left:
            1rem !important;

        padding-right:
            1rem !important;
    }

    .hero {
        padding:
            22px;
    }
}

@media (max-width: 600px) {

    .hero {
        border-radius:
            18px;

        padding:
            18px;
    }

    .hero-title {
        font-size:
            2.4rem;
    }
}

</style>
"""

st.markdown(
    APP_CSS,
    unsafe_allow_html=True,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/aasist/aasist.onnx"

AUDIO_PATH = "data/sample_audio/test_pcm.wav"

TARGET_LENGTH = 64600


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def load_model():

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=[
            "CPUExecutionProvider"
        ],
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

    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1,
        )

    max_value = np.max(
        np.abs(audio)
    )

    if max_value > 0:

        audio = (
            audio / max_value
        )

    return audio, sample_rate


# ============================================================
# AUDIO PREPARATION
# ============================================================

def prepare_audio(audio):

    if len(audio) >= TARGET_LENGTH:

        audio = audio[
            :TARGET_LENGTH
        ]

    else:

        padding = (
            TARGET_LENGTH
            - len(audio)
        )

        audio = np.pad(
            audio,
            (
                0,
                padding,
            ),
        )

    return audio.astype(
        np.float32
    )


# ============================================================
# TEMP AUDIO
# ============================================================

def save_uploaded_audio_temp(
    file_bytes,
    prefix,
):

    temp = tempfile.NamedTemporaryFile(
        suffix=".wav",
        prefix=prefix,
        delete=False,
    )

    try:

        temp.write(
            file_bytes
        )

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

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    model_input = (
        audio.reshape(
            1,
            TARGET_LENGTH,
        )
    )

    outputs = session.run(
        None,
        {
            input_name:
                model_input
        },
    )

    logits = outputs[0]

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

    return (
        logits,
        probabilities,
    )


# ============================================================
# FRAUD PHRASE INTELLIGENCE
# ============================================================

def analyze_fraud_context(text):

    text = (
        text
        .lower()
        .strip()
    )

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

    for (
        signal_name,
        phrases,
        weight,
    ) in rules:

        matched = []

        for phrase in phrases:

            if phrase in text:

                matched.append(
                    phrase
                )

        if matched:

            signals.append(
                signal_name
            )

            matched_phrases.extend(
                matched
            )

            risk_score += weight

    matched_phrases = list(
        dict.fromkeys(
            matched_phrases
        )
    )

    return {
        "risk_score":
            float(
                min(
                    risk_score,
                    100,
                )
            ),

        "signals":
            signals,

        "matched_phrases":
            matched_phrases,
    }
# ============================================================
# LIVE SECURITY DECISION OVERRIDE
# ============================================================

try:
    live_security_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "live",
        "live_status.json",
    )

    if os.path.exists(live_security_path):
        import json

        with open(
            live_security_path,
            "r",
            encoding="utf-8",
        ) as live_security_file:
            live_security_data = json.load(live_security_file)

        live_security_risk = str(
            live_security_data.get(
                "risk_level",
                "",
            )
        ).upper()

        if live_security_risk in [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL",
        ]:
            risk_level = live_security_risk

except Exception:
    pass

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

    amount_risk = 0

    if transaction_amount >= 10000000:

        amount_risk = 35

    elif transaction_amount >= 1000000:

        amount_risk = 25

    elif transaction_amount >= 100000:

        amount_risk = 15

    elif transaction_amount >= 10000:

        amount_risk = 5

    beneficiary_risk = (
        20
        if new_beneficiary
        else 0
    )

    time_risk = (
        10
        if unusual_time
        else 0
    )

    location_risk = (
        15
        if location_change
        else 0
    )

    velocity_risk = (
        15
        if multiple_recent_transactions
        else 0
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

    return float(
        total_context_risk
    )


# ============================================================
# CONTEXT + FRAUD FUSION
# ============================================================

def combine_context_risk(
    transaction_risk,
    fraud_phrase_risk,
):

    combined_risk = (
        0.70
        * transaction_risk
        +
        0.30
        * fraud_phrase_risk
    )

    return float(
        min(
            combined_risk,
            100,
        )
    )


# ============================================================
# FINAL RISK ENGINE
# ============================================================

def calculate_final_risk(
    spoof_probability,
    context_risk,
    fraud_risk=0.0,
):
    spoof_pct = float(spoof_probability) * 100.0
    context_pct = float(context_risk)
    fraud_pct = float(fraud_risk)

    # ---------------------------------------------------------
    # Base fused score
    # ---------------------------------------------------------
    final_score = (
        spoof_pct * 0.60
        + context_pct * 0.20
        + fraud_pct * 0.20
    )

    final_score = min(max(final_score, 0.0), 100.0)

    # ---------------------------------------------------------
    # SIGNAL-BASED SECURITY FLOORS
    # One strong signal must not disappear inside averaging.
    # ---------------------------------------------------------

    # CRITICAL
    if spoof_pct >= 90:
        final_score = max(final_score, 80.0)
        risk_level = "CRITICAL"
        recommendation = (
            "Strong voice-spoofing signal detected. "
            "Block sensitive action and perform independent verification."
        )

    elif fraud_pct >= 70:
        final_score = max(final_score, 80.0)
        risk_level = "CRITICAL"
        recommendation = (
            "Critical fraud intent detected. "
            "Freeze sensitive action and require strong verification."
        )

    elif context_pct >= 80:
        final_score = max(final_score, 80.0)
        risk_level = "CRITICAL"
        recommendation = (
            "Critical transaction/context risk detected. "
            "Freeze sensitive action and verify independently."
        )

    # HIGH
    elif spoof_pct >= 70:
        final_score = max(final_score, 60.0)
        risk_level = "HIGH"
        recommendation = (
            "High voice impersonation risk detected. "
            "Stop sensitive action and perform secondary verification."
        )

    elif fraud_pct >= 45:
        final_score = max(final_score, 60.0)
        risk_level = "HIGH"
        recommendation = (
            "High fraud-intent signal detected. "
            "Stop sensitive action and require verification."
        )

    elif context_pct >= 60:
        final_score = max(final_score, 60.0)
        risk_level = "HIGH"
        recommendation = (
            "High transaction/context risk detected. "
            "Stop sensitive action and verify."
        )

    # SUSPICIOUS / MEDIUM
    elif spoof_pct >= 40:
        final_score = max(final_score, 35.0)
        risk_level = "MEDIUM"
        recommendation = (
            "Suspicious voice characteristics detected. "
            "Proceed only after additional verification."
        )

    elif fraud_pct >= 30:
        final_score = max(final_score, 35.0)
        risk_level = "MEDIUM"
        recommendation = (
            "Suspicious fraud-intent language detected. "
            "Perform additional verification."
        )

    elif context_pct >= 35:
        final_score = max(final_score, 35.0)
        risk_level = "MEDIUM"
        recommendation = (
            "Suspicious transaction/context signals detected. "
            "Perform additional verification."
        )

    # LOW
    else:
        risk_level = "LOW"
        recommendation = (
            "No strong spoofing, fraud or contextual risk signal detected. "
            "Continue normal verification."
        )

    return (
        spoof_pct,
        round(final_score, 2),
        risk_level,
        recommendation,
    )


# ============================================================
# SENTINEL AI COPILOT
# ============================================================

def get_copilot_reply(question):

    q = (
        question
        .lower()
        .strip()
    )

    if any(
        word in q
        for word in [
            "kaise kaam",
            "how does",
            "working",
            "work",
            "workflow",
            "flow",
            "app kya",
            "kya karta",
        ]
    ):

        return (
            "ShwaraShield ka workflow hai: "
            "Voice Input → AASIST + ONNX voice authenticity analysis → "
            "transaction/context risk → fraud phrase intelligence → "
            "voice + context risk fusion → "
            "LOW/MEDIUM/HIGH/CRITICAL decision → "
            "high-risk par Adaptive Challenge Verification."
        )

    if any(
        word in q
        for word in [
            "aasist",
            "onnx",
            "model",
            "ai engine",
            "voice detection",
        ]
    ):

        return (
            "AI Engine AASIST anti-spoofing model aur ONNX Runtime "
            "use karta hai. Uploaded audio ko fixed model input "
            "length ke liye prepare karke inference chalaya jata hai "
            "aur spoof/bonafide probabilities nikali jati hain."
        )

    if any(
        word in q
        for word in [
            "risk",
            "score",
            "high",
            "critical",
            "medium",
            "low",
        ]
    ):

        return (
            "Risk Engine voice risk aur context risk ko fuse karta hai. "
            "LOW result normal verification ke saath continue hota hai, "
            "MEDIUM par additional verification, HIGH par STOP & VERIFY, "
            "aur CRITICAL par BLOCK ACTION recommend kiya jata hai."
        )

    if any(
        word in q
        for word in [
            "context",
            "transaction",
            "beneficiary",
            "amount",
            "location",
            "time",
            "multiple",
        ]
    ):

        return (
            "Context Risk requested action, transaction amount, "
            "new/unknown beneficiary, unusual transaction time, "
            "unusual location/device change, aur multiple recent "
            "transactions jaise signals se calculate hota hai."
        )

    if any(
        word in q
        for word in [
            "phrase",
            "language",
            "otp",
            "urgent",
            "jaldi",
            "fraud",
        ]
    ):

        return (
            "Fraud Phrase Intelligence entered Hindi, English aur "
            "Hinglish text mein urgency, OTP/security credential "
            "requests, money transfer requests aur confidential-"
            "information patterns ko check karta hai."
        )

    if any(
        word in q
        for word in [
            "challenge",
            "verification phrase",
            "adaptive",
        ]
    ):

        return (
            "Adaptive Challenge Verification sirf HIGH ya CRITICAL "
            "risk par activate hota hai. App ek random phrase generate "
            "karta hai aur prototype mein caller se us phrase ko repeat "
            "karne ke liye kaha jata hai. Future version mein repeated "
            "response par speaker consistency aur liveness checks "
            "add kiye ja sakte hain."
        )

    if any(
        word in q
        for word in [
            "privacy",
            "audio store",
            "retention",
            "data",
        ]
    ):

        return (
            "Current prototype local/edge inference aur minimal audio "
            "retention approach dikhata hai. Production deployment "
            "mein retention, consent aur access controls explicitly "
            "enforce karne honge."
        )

    if any(
        word in q
        for word in [
            "result",
            "latest",
            "current",
            "abhi",
            "final decision",
        ]
    ):

        latest = st.session_state.get(
            "latest_result"
        )

        if latest:

            return (
                f"Latest available analysis: "
                f"Voice Risk {latest['voice_risk']:.2f}%, "
                f"Context Risk {latest['context_risk']:.2f}%, "
                f"Final Risk {latest['final_score']:.2f}, "
                f"Level {latest['risk_level']}."
            )

        return (
            "Abhi koi analysis result available nahi hai. "
            "Pehle voice sample upload karke "
            "Analyze Voice & Transaction run karo."
        )

    return (
        "Main ShwaraShield AI Copilot hoon. "
        "Main is app ka workflow, AASIST + ONNX, risk score, "
        "transaction/context signals, fraud phrase intelligence, "
        "privacy, Adaptive Challenge Verification aur current "
        "result explain kar sakta hoon."
    )


# ============================================================
# DASHBOARD HEADER
# ============================================================

title_col, copilot_col, status_col = st.columns(
    [6, 1.4, 2.0]
)


with title_col:

    st.title(
        "🛡️ ShwaraShield"
    )

    st.subheader(
        "AI-Powered Real-Time Voice Cloning & Impersonation Defense"
    )

    st.caption(
        "AI-powered voice integrity verification "
        "with context-aware fraud prevention."
    )


with copilot_col:

    st.write("")

    with st.popover(
        "🤖",
        use_container_width=True,
    ):

        st.markdown(
            "### 🤖 Sentinel AI Copilot"
        )

        st.caption(
            "Ask about ShwaraShield, AASIST, "
            "risk scores, fraud signals or verification."
        )

        if (
            "copilot_messages"
            not in st.session_state
        ):

            st.session_state[
                "copilot_messages"
            ] = [
                {
                    "role": "assistant",
                    "content": (
                        "Namaste! Main Sentinel AI Copilot hoon. "
                        "Main ShwaraShield ke workflow, AASIST, "
                        "risk engine, fraud intelligence aur "
                        "security decisions explain kar sakta hoon."
                    ),
                }
            ]

        for message in st.session_state[
            "copilot_messages"
        ][-6:]:

            if message["role"] == "user":

                st.markdown(
                    f"**You:** {message['content']}"
                )

            else:

                st.markdown(
                    f"**Sentinel:** {message['content']}"
                )

        copilot_question = st.text_input(
            "Ask Sentinel",
            key="copilot_question_input",
            placeholder="e.g. How does AASIST work?",
            label_visibility="collapsed",
        )

        if st.button(
            "Send",
            key="copilot_send",
            use_container_width=True,
        ):

            if copilot_question.strip():

                st.session_state[
                    "copilot_messages"
                ].append(
                    {
                        "role":
                            "user",

                        "content":
                            copilot_question.strip(),
                    }
                )

                reply = get_copilot_reply(
                    copilot_question
                )

                st.session_state[
                    "copilot_messages"
                ].append(
                    {
                        "role":
                            "assistant",

                        "content":
                            reply,
                    }
                )

                st.rerun()


with status_col:

    st.write("")

    st.markdown(
        '<div style="text-align:right;">'
        '<span class="live-pill">'
        '<span class="live-dot"></span>'
        'Protected'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "AI monitoring active"
    )


st.divider()


# ============================================================
# SYSTEM OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-kicker">'
    'System Overview'
    '</div>',
    unsafe_allow_html=True,
)


ov1, ov2, ov3 = st.columns(3)


with ov1:

    with st.container(
        border=True
    ):

        st.markdown(
            "### 🎙️ Voice Integrity"
        )

        st.write(
            "AASIST anti-spoofing analysis checks "
            "the authenticity of the caller voice."
        )

        st.markdown(
            '<div class="engine-ready">'
            'AASIST ENGINE READY'
            '</div>',
            unsafe_allow_html=True,
        )


with ov2:

    with st.container(
        border=True
    ):

        st.markdown(
            "### 🧠 Fraud Intelligence"
        )

        st.write(
            "Hindi, English and Hinglish fraud patterns "
            "are analyzed for suspicious intent."
        )

        st.markdown(
            '<div class="engine-ready">'
            'LANGUAGE ENGINE READY'
            '</div>',
            unsafe_allow_html=True,
        )


with ov3:

    with st.container(
        border=True
    ):

        st.markdown(
            "### 🔐 Security Layer"
        )

        st.write(
            "Voice, transaction and contextual signals "
            "are fused into one security decision."
        )

        st.markdown(
            '<div class="engine-ready">'
            'RISK ENGINE READY'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# LIVE PROTECTION STATUS
# ============================================================

try:

    _ss_fragment = st.fragment

except AttributeError:

    _ss_fragment = None


def _render_live_protection():

    st.markdown(
        '<div class="section-kicker">'
        'Live Protection'
        '</div>',
        unsafe_allow_html=True,
    )

    live_col1, live_col2, live_col3, live_col4 = (
        st.columns(4)
    )

    live_status_path = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "data",
        "live",
        "live_status.json",
    )

    live_data = None

    try:

        if os.path.exists(
            live_status_path
        ):

            with open(
                live_status_path,
                "r",
                encoding="utf-8",
            ) as live_file:

                live_data = json.load(
                    live_file
                )

    except Exception:

        live_data = None

    if live_data:

        live_risk = float(
            live_data.get(
                "acoustic_risk",
                0.0,
            )
        )

        live_spoof = (
            float(
                live_data.get(
                    "spoof_probability",
                    0.0,
                )
            )
            * 100
        )

        live_genuine = (
            float(
                live_data.get(
                    "genuine_probability",
                    0.0,
                )
            )
            * 100
        )

        live_state = str(
            live_data.get(
                "status",
                "MONITORING",
            )
        )

        participant = str(
            live_data.get(
                "participant",
                "Unknown",
            )
        )

    else:

        live_risk = 0.0
        live_spoof = 0.0
        live_genuine = 0.0

        live_state = "WAITING"

        participant = (
            "Waiting for LiveKit audio"
        )

    with live_col1:

        st.metric(
            "Live Status",
            live_state,
        )

    with live_col2:

        st.metric(
            "Acoustic Risk",
            f"{live_risk:.2f}%",
        )

    with live_col3:

        st.metric(
            "Spoof Probability",
            f"{live_spoof:.2f}%",
        )

    with live_col4:

        st.metric(
            "Bonafide Probability",
            f"{live_genuine:.2f}%",
        )

    st.caption(
        f"LiveKit participant: {participant} "
        "• Auto-refresh: 2s"
    )


if _ss_fragment is not None:

    _render_live_protection = (
        _ss_fragment(
            run_every="2s"
        )(
            _render_live_protection
        )
    )


_render_live_protection()


# ============================================================
# LIVE CALL PROTECTION
# ============================================================

st.markdown(
    '<div class="section-kicker">'
    'Live Call Protection'
    '</div>',
    unsafe_allow_html=True,
)


if (
    LIVEKIT_UI_AVAILABLE
    and render_livekit_panel is not None
):

    render_livekit_panel()

else:

    st.error(
        "LiveKit integration could not be loaded. "
        + LIVEKIT_UI_ERROR
    )


# ============================================================
# TRANSACTION CONTEXT
# ============================================================

st.markdown(
    '<div class="section-kicker">'
    'Transaction Context'
    '</div>',
    unsafe_allow_html=True,
)


ctx1, ctx2 = st.columns(2)


with ctx1:

    with st.container(
        border=True
    ):

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


with ctx2:

    with st.container(
        border=True
    ):

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


sig1, sig2, sig3, sig4 = st.columns(4)


with sig1:

    new_beneficiary = st.checkbox(
        "👤 New / unknown beneficiary",
        value=False,
    )


with sig2:

    unusual_time = st.checkbox(
        "🌙 Unusual transaction time",
        value=False,
    )


with sig3:

    location_change = st.checkbox(
        "📍 Unusual location / device change",
        value=False,
    )


with sig4:

    multiple_recent_transactions = st.checkbox(
        "🔁 Multiple recent transactions",
        value=False,
    )


# ============================================================
# VOICE + SPEAKER INPUT
# ============================================================

st.markdown(
    '<div class="section-kicker">'
    'Voice Security'
    '</div>',
    unsafe_allow_html=True,
)


voice_col, speaker_col = st.columns(2)


with voice_col:

    with st.container(
        border=True
    ):

        st.markdown(
            "### 🎙️ Voice Input"
        )

        uploaded_file = st.file_uploader(
            "Upload voice sample",
            type=["wav"],
            help=(
                "Upload a WAV file for "
                "voice authenticity analysis."
            ),
        )


with speaker_col:

    with st.container(
        border=True
    ):

        st.markdown(
            "### 👤 Speaker Consistency Check"
        )

        registered_voice_file = st.file_uploader(
            "Upload registered genuine voice sample",
            type=["wav"],
            key="registered_voice_sample",
            help=(
                "Optional trusted voice sample used "
                "to compare the current caller voice."
            ),
        )


# ============================================================
# CALLER TRANSCRIPT
# ============================================================

with st.container(
    border=True
):

    st.markdown(
        "### 💬 Caller Message / Transcript"
    )

    caller_text = st.text_area(
        "Enter caller's spoken message or transcript",
        placeholder=(
            "Example: Main bank se bol raha hoon. "
            "Jaldi paise transfer karo aur "
            "kisi ko mat batana."
        ),
        height=120,
        label_visibility="collapsed",
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


if caller_text and caller_text.strip():

    with st.container(
        border=True
    ):

        st.markdown(
            "### 🧠 Fraud Context Intelligence"
        )

        f1, f2 = st.columns(2)

        with f1:

            st.metric(
                "Fraud Phrase Risk",
                f"{fraud_phrase_risk:.0f}%",
            )

        with f2:

            st.metric(
                "Detected Signals",
                len(fraud_signals),
            )

        if fraud_signals:

            for signal in fraud_signals:

                st.warning(
                    f"⚠️ {signal}"
                )

        else:

            st.success(
                "No high-risk fraud phrases detected."
            )

        if matched_phrases:

            st.caption(
                "Matched phrases"
            )

            st.write(
                " • ".join(
                    matched_phrases
                )
            )


st.divider()


# ============================================================
# ANALYZE
# ============================================================

if st.button(
    "🔍 ANALYZE VOICE & TRANSACTION",
    use_container_width=True,
):

    try:

        # ====================================================
        # LOAD MODEL
        # ====================================================

        with st.spinner(
            "Loading AI detection model..."
        ):

            session = load_model()


        # ====================================================
        # LOAD AUDIO
        # ====================================================

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


        # ====================================================
        # SAMPLE RATE
        # ====================================================

        if sample_rate != 16000:

            st.warning(
                f"Audio sample rate is "
                f"{sample_rate} Hz. "
                "The AASIST prototype expects "
                "16 kHz audio."
            )


        # ====================================================
        # PREPARE AUDIO
        # ====================================================

        prepared_audio = (
            prepare_audio(
                audio
            )
        )


        # ====================================================
        # AASIST
        # ====================================================

        with st.spinner(
            "Running AI voice authenticity analysis..."
        ):

            logits, probabilities = (
                detect_voice(
                    session,
                    prepared_audio,
                )
            )


        # ====================================================
        # CLASS MAPPING
        # ====================================================

        spoof_probability = float(
            probabilities[0][0]
        )

        genuine_probability = float(
            probabilities[0][1]
        )


        # ====================================================
        # SPEAKER VERIFICATION
        # ====================================================

        speaker_result = None

        speaker_error = None

        temp_paths = []


        if registered_voice_file is not None:

            if not SPEAKER_VERIFICATION_AVAILABLE:

                speaker_error = (
                    "speaker_verification.py "
                    "could not be loaded."
                )

            else:

                try:

                    with st.spinner(
                        "Checking speaker consistency..."
                    ):

                        registered_path = (
                            save_uploaded_audio_temp(
                                registered_voice_file.getvalue(),
                                "ShwaraShield_registered_",
                            )
                        )

                        current_path = (
                            save_uploaded_audio_temp(
                                file_bytes,
                                "ShwaraShield_current_",
                            )
                        )

                        temp_paths = [
                            registered_path,
                            current_path,
                        ]

                        speaker_result = (
                            verify_speaker(
                                registered_path,
                                current_path,
                            )
                        )

                except Exception as error:

                    speaker_error = str(
                        error
                    )

                finally:

                    for temp_path in temp_paths:

                        try:

                            if os.path.exists(
                                temp_path
                            ):

                                os.remove(
                                    temp_path
                                )

                        except OSError:

                            pass


        # ====================================================
        # TRANSACTION RISK
        # ====================================================

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


        # ====================================================
        # CONTEXT RISK
        # ====================================================

        context_risk = (
            combine_context_risk(
                transaction_context_risk,
                fraud_phrase_risk,
            )
        )


        # ====================================================
        # FINAL RISK
        # ====================================================

        (
            voice_risk,
            final_score,
            risk_level,
            recommendation,
        ) = calculate_final_risk(
            spoof_probability,
            context_risk,
        )


        # ====================================================
        # SAVE RESULT
        # ====================================================

        st.session_state[
            "latest_result"
        ] = {

            "voice_risk":
                voice_risk,

            "context_risk":
                context_risk,

            "final_score":
                final_score,

            "risk_level":
                risk_level,

            "speaker_similarity":
                (
                    float(
                        speaker_result.get(
                            "similarity",
                            0.0,
                        )
                    )
                    if speaker_result is not None
                    else None
                ),

            "speaker_status":
                (
                    speaker_result.get(
                        "status",
                        "UNKNOWN",
                    )
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


        # ====================================================
        # METRICS
        # ====================================================

        col1, col2, col3, col4, col5 = (
            st.columns(5)
        )


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

                speaker_match_text = (
                    "Not checked"
                )

            st.metric(
                "👤 Speaker Match",
                speaker_match_text,
            )


        st.divider()


        # ====================================================
        # SPEAKER RESULT
        # ====================================================

        if speaker_result is not None:

            st.subheader(
                "👤 Speaker Consistency Result"
            )

            speaker_col1, speaker_col2, speaker_col3 = (
                st.columns(3)
            )

            similarity = float(
                speaker_result.get(
                    "similarity",
                    0.0,
                )
            )

            speaker_status = (
                speaker_result.get(
                    "status",
                    "UNKNOWN",
                )
            )

            speaker_risk = (
                speaker_result.get(
                    "risk",
                    "UNKNOWN",
                )
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


            if (
                str(
                    speaker_risk
                ).upper()
                == "HIGH"
            ):

                st.error(
                    "⚠️ Speaker mismatch detected. "
                    "Use independent verification."
                )

            elif (
                str(
                    speaker_risk
                ).upper()
                == "MEDIUM"
            ):

                st.warning(
                    "⚠️ Speaker match needs "
                    "additional verification."
                )

            else:

                st.success(
                    "✓ Current voice is consistent "
                    "with the registered genuine voice."
                )


        elif speaker_error:

            st.warning(
                "Speaker Consistency Check could not "
                "be completed: "
                + speaker_error
            )


        # ====================================================
        # RISK LEVEL
        # ====================================================
                # Persistent transaction protection state
        transaction_frozen = st.session_state.get(
            "transaction_frozen",
            False,
        )

        if risk_level == "CRITICAL":
            transaction_frozen = True
            st.session_state["transaction_frozen"] = True

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


        # ====================================================
        # FINAL RISK GAUGE
        # ====================================================

        st.subheader(
            "🛡️ Final Impersonation Risk"
        )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=final_score,
                title={
                    "text":
                        "Risk Score"
                },
                gauge={
                    "axis": {
                        "range":
                            [0, 100]
                    },

                    "bar": {
                        "thickness":
                            0.75
                    },

                    "steps": [

                        {
                            "range":
                                [0, 35]
                        },

                        {
                            "range":
                                [35, 60]
                        },

                        {
                            "range":
                                [60, 80]
                        },

                        {
                            "range":
                                [80, 100]
                        },

                    ],
                },
            )
        )

        fig.update_layout(
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


        # ====================================================
        # RISK EXPLANATION
        # ====================================================

        st.subheader(
            "🧠 Why did ShwaraShield raise this risk?"
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


        # ====================================================
        # FRAUD LANGUAGE EXPLANATION
        # ====================================================

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


        # ====================================================
        # ADAPTIVE CHALLENGE
        # ====================================================

        if risk_level in [
            "HIGH",
            "CRITICAL",
        ]:

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

            challenge_phrase = (
                random.choice(
                    challenge_phrases
                )
            )

            st.warning(
                "⚠️ High-risk voice detected. "
                "Ask the caller to repeat "
                "the verification phrase."
            )

            st.info(
                f"🔐 Verification Phrase: "
                f"**{challenge_phrase}**"
            )

            st.caption(
                "Prototype security challenge. "
                "A production system would compare the repeated "
                "response using speaker consistency and liveness checks."
            )


        # ====================================================
        # SECURITY DECISION
        # ====================================================

        st.divider()

        st.subheader(
            "🔐 Security Decision"
        )


        # ====================================================
        # CRITICAL
        # ====================================================

        if risk_level == "CRITICAL":

            st.error(
                "🚨 BLOCK ACTION"
            )

            st.error(
                "🔒 TRANSACTION FROZEN — "
                "VERIFICATION REQUIRED"
            )

            st.info(
                "ShwaraShield has temporarily blocked the "
                "sensitive transaction in this prototype because "
                "critical voice-fraud risk was detected."
            )

            col_a, col_b = st.columns(2)

            with col_a:

                st.metric(
                    "Protection Action",
                    "FROZEN",
                )

            with col_b:

                st.metric(
                    "Verification",
                    "REQUIRED",
                )

            st.caption(
                "Prototype simulation only — production deployment "
                "would require authorized banking/transaction APIs."
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


            # =================================================
            # INDEPENDENT VERIFICATION
            # =================================================

            st.divider()

            st.subheader(
                "🛡️ Independent Verification"
            )

            st.warning(
                "Transaction remains frozen until "
                "the caller is independently verified."
            )

            verification_method = st.selectbox(
                "Verification Method",
                [
                    "Select verification method",
                    "Trusted-number callback",
                    "MFA / OTP verification",
                    "Known-contact confirmation",
                    "Supervisor / security verification",
                ],
                key="verification_method",
            )

            verification_code = st.text_input(
                "Verification Reference",
                placeholder=(
                    "Enter verification reference"
                ),
                key="verification_reference",
            )

            if st.button(
                "✅ Complete Verification",
                key="complete_verification",
            ):

                if (
                    verification_method
                    !=
                    "Select verification method"
                    and
                    verification_code.strip()
                ):

                    st.session_state[
                        "verification_completed"
                    ] = True

                    st.success(
                        "✅ Independent verification "
                        "completed successfully."
                    )

                    st.info(
                        "🔓 Prototype protection state: "
                        "transaction can now be released."
                    )

                else:

                    st.session_state[
                        "verification_completed"
                    ] = False

                    st.error(
                        "⚠️ Select a verification method "
                        "and enter a verification reference."
                    )


            if st.session_state.get(
                "verification_completed",
                False,
            ):

                st.success(
                    "🔓 VERIFICATION PASSED — "
                    "PROTECTION RELEASED"
                )


        # ====================================================
        # HIGH
        # ====================================================

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


        # ====================================================
        # MEDIUM
        # ====================================================

        elif risk_level == "MEDIUM":

            st.warning(
                "🟠 CAUTION"
            )

            st.write(
                "Perform additional verification before "
                "continuing with the requested action."
            )


        # ====================================================
        # LOW
        # ====================================================

        else:

            st.success(
                "🟢 ALLOW WITH NORMAL VERIFICATION"
            )

            st.write(
                "The current prototype assessment is low-risk, "
                "but standard verification should still apply."
            )


        # ====================================================
        # TECHNICAL DETAILS
        # ====================================================

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


    # ========================================================
    # ANALYSIS ERROR
    # ========================================================

    except Exception as error:

        st.error(
            "Analysis failed."
        )

        st.exception(
            error
        )
