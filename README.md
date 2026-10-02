# 🛡️ SwaraShield

### AI-Powered Caller Protection Against Voice Cloning, Scams & Social Engineering

> **SwaraShield is an open-source research and development project focused on building an intelligent caller-protection layer that can detect suspicious voice activity, cloned voices, and scam-like conversations in real time.**

The long-term vision is to build a privacy-conscious, AI-powered alternative to traditional caller identification systems — moving from simply **"Who is calling?"** to **"Is this conversation safe?"**

---

## 🚨 Why SwaraShield?

Traditional caller-ID applications primarily answer:

> **Who is calling me?**

Modern scams can involve:

- 🎭 AI-generated / cloned voices
- 💰 Urgent money-transfer requests
- 🔐 Requests for OTPs, passwords, or sensitive information
- 🚨 Threats and social engineering
- 📞 Impersonation of friends, family, banks, police, or organizations

A known phone number does not necessarily mean a safe conversation.

**SwaraShield aims to add an AI-powered security layer to the conversation itself.**

---

# 🎯 Vision

```text
Incoming / Ongoing Call
        ↓
Audio & Conversation Analysis
        ↓
Voice Authenticity Detection
        ↓
Scam / Fraud Detection
        ↓
Risk Engine
        ↓
┌─────────────────────────────┐
│        SAFETY DECISION      │
├─────────────────────────────┤
│ 🟢 SAFE                     │
│ 🟡 SUSPICIOUS               │
│ 🟠 HIGH RISK                │
│ 🔴 CRITICAL                 │
└─────────────────────────────┘
        ↓
User Warning / Protection
```

The objective is not simply to identify a caller.

The objective is to **protect the person receiving the call.**

---

# ⚙️ Current Prototype

SwaraShield currently operates as a **controlled browser/VoIP prototype**.

```text
Browser Microphone
       ↓
LiveKit / WebRTC
       ↓
SwaraShield Interceptor
       ↓
Audio Chunk Processing
       ↓
AASIST
       ↓
Voice Spoof / Authenticity Analysis
       ↓
Risk Engine
       ↓
Real-Time Security Status
```

The prototype does **not** claim to silently intercept arbitrary cellular calls.

---

# 🧠 Core Components

## 1. 🎙️ Real-Time Audio Processing

Audio is captured from the controlled call environment and processed in chunks.

The pipeline includes:

- Audio extraction
- Resampling
- Mono conversion
- Normalization
- Chunk-based inference

Current target sample rate:

```text
16 kHz
```

## 2. 🧬 Voice Spoof Detection

SwaraShield currently experiments with **AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal graph attention networks)** for detecting spoofed or synthetic speech.

> ⚠️ Voice-spoof detection is an active research area. Model predictions are probabilistic and should not be treated as perfect proof that a voice is AI-generated.

## 3. 🧠 Scam / Fraud Detection

The future pipeline can combine voice analysis with conversational analysis.

Potential signals include:

- Money-transfer requests
- OTP requests
- Account/payment requests
- Threats
- Urgency
- Impersonation
- Suspicious instructions
- Social-engineering patterns

---

# 🔥 Risk Engine

SwaraShield is designed around **multi-signal risk fusion** rather than relying on one AI model.

```text
Voice Spoof Risk
       +
Conversation/Fraud Risk
       +
Behavioral Signals
       +
Speaker Verification
       ↓
   Risk Engine
       ↓
Final Risk Score
```

The architecture is designed so additional detectors can be added without redesigning the entire system.

---

# 🏗️ Architecture

```text
                    ┌─────────────────┐
                    │ Browser / VoIP  │
                    │   Microphone    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    LiveKit      │
                    │   WebRTC Audio  │
                    └────────┬────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ SwaraShield         │
                  │ Audio Interceptor   │
                  └──────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       ┌────────────┐ ┌────────────┐ ┌──────────────┐
       │   AASIST   │ │   Speaker  │ │ Conversation │
       │ Anti-Spoof │ │ Verification│ │   Analysis   │
       └──────┬─────┘ └──────┬─────┘ └───────┬──────┘
              │              │               │
              └──────────────┼───────────────┘
                             ▼
                    ┌─────────────────┐
                    │   Risk Engine   │
                    └────────┬────────┘
                             │
                             ▼
                 ┌──────────────────────┐
                 │   Security Decision  │
                 └──────────┬───────────┘
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
          🟢 SAFE       🟡 WARNING      🔴 CRITICAL
```

See [`docs/architecture.md`](docs/architecture.md) for more detail.

---

# 🔮 Long-Term Vision

### From:

> **Caller Identification**

### To:

> **Caller Security Intelligence**

A future version could combine:

```text
Caller Identity
      +
Voice Authenticity
      +
Conversation Intelligence
      +
Fraud Detection
      +
Behavioral Analysis
      +
Threat Intelligence
      ↓
AI Caller Security Layer
```

Possible capabilities:

- Real-time scam warnings
- AI voice-clone detection
- Speaker verification
- Financial scam detection
- Social-engineering detection
- Suspicious-link detection
- Caller reputation
- Community threat intelligence
- Privacy-preserving on-device AI
- Explainable risk alerts
- User-controlled protection actions

---

# 🛣️ Roadmap

## Phase 1 — Prototype ✅

- [x] LiveKit integration
- [x] Real-time audio subscription
- [x] Audio chunk processing
- [x] 16 kHz audio pipeline
- [x] AASIST integration
- [x] Spoof/genuine inference
- [x] Initial risk engine
- [x] Live security status
- [x] Controlled browser/VoIP demonstration

## Phase 2 — Intelligence 🔄

- [ ] Improve voice-spoof detection stability
- [ ] Add speaker verification
- [ ] Add speech-to-text
- [ ] Add scam/fraud NLP
- [ ] Improve risk fusion
- [ ] Reduce false positives
- [ ] Add explainable alerts
- [ ] Build stronger evaluation datasets

## Phase 3 — Real-Time Protection 🚀

- [ ] Production-grade streaming pipeline
- [ ] Low-latency inference
- [ ] Better noise robustness
- [ ] Multi-language support
- [ ] Advanced scam detection
- [ ] Threat-intelligence integration
- [ ] Privacy-preserving processing

## Phase 4 — Real-World Product 🌍

- [ ] Android implementation
- [ ] iOS research
- [ ] Telecom/VoIP integrations
- [ ] Production security architecture
- [ ] Privacy and consent framework
- [ ] Large-scale evaluation
- [ ] Enterprise / telecom partnerships

---

# 📁 Repository Structure

```text
SwaraShield/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── .gitignore
├── .env.example
├── requirements.txt
│
├── src/
│   ├── interceptor/
│   ├── audio/
│   ├── models/
│   ├── risk_engine/
│   └── utils/
│
├── models/
├── data/
├── tests/
├── app/
├── docs/
└── screenshots/
```

---

# 🧑‍💻 Quick Start

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/SwaraShield.git
cd SwaraShield
```

### 2. Create virtual environment

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Fill in the required values locally.

### 5. Run tests

```bash
pytest
```

### 6. Run the prototype

The exact application entry point depends on the current prototype integration. Keep secrets in `.env` and follow the project documentation.

---

# 🔐 Environment Variables

Never commit `.env`.

Example:

```env
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
SWARASHIELD_AASIST_MODEL=models/aasist/aasist.onnx
```

---

# 🤝 Contributing

SwaraShield is intentionally being developed as an open-source project.

We welcome contributions in:

- AI / ML
- Speech AI
- Audio processing
- Deepfake detection
- Cybersecurity
- Fraud detection
- NLP
- Android / iOS
- Backend
- WebRTC / real-time systems
- UI/UX
- Research and evaluation
- Documentation

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before submitting a pull request.

---

# 🧪 Research & Evaluation

SwaraShield should be evaluated as a security/ML system, not only by whether a demo works.

Important metrics include:

- False-positive rate
- False-negative rate
- Equal Error Rate (EER)
- ROC-AUC where appropriate
- Detection latency
- CPU/GPU usage
- Robustness to noise
- Robustness to codecs/compression
- Performance across speakers and languages
- Calibration of risk scores

See [`docs/research.md`](docs/research.md).

---

# ⚠️ Current Limitations

SwaraShield is currently a **research/prototype project**.

Current limitations include:

- Voice-spoof models can produce unstable predictions.
- Background noise can affect inference.
- False positives and false negatives are possible.
- The prototype operates in a controlled browser/VoIP environment.
- It does not automatically intercept arbitrary cellular calls.
- Production banking actions are not implemented.
- More datasets and real-world testing are required.
- Privacy, security, regulatory, and platform constraints must be addressed before deployment.

---

# 🔒 Privacy & Responsible Use

Privacy is a core consideration.

The long-term architecture should prioritize:

- Data minimization
- User consent
- Encryption
- Secure processing
- On-device inference where practical
- Limited data retention
- Transparent AI decisions
- User-controlled alerts
- Protection of sensitive conversations

Do not use this project to unlawfully record, intercept, monitor, or analyze private communications.

Always follow applicable laws, platform policies, and consent requirements.

---

# 🌟 Join the Project

SwaraShield is not finished.

**That's the point.**

The goal is to build this together with researchers, developers, cybersecurity enthusiasts, AI engineers, and people who want to work on the next generation of caller security.

⭐ Star the repository  
🍴 Fork the project  
🐛 Open an issue  
💡 Suggest an idea  
🔧 Submit a pull request  
🤝 Start a discussion

---

## 🚀 The Vision

> **Don't just identify the caller. Protect the conversation.**

**SwaraShield — Building an intelligent security layer for voice communication.**

---

### Built with ❤️ for AI, Cybersecurity & Open Source
