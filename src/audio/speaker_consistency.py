import numpy as np


# ============================================================
# VOICESENTINEL - IMPROVED SPEAKER CONSISTENCY ENGINE
# ============================================================

SAMPLE_RATE = 16000
FRAME_SIZE = 400
HOP_SIZE = 160
NUM_FILTERS = 20
NUM_CEPS = 13


def normalize_audio(audio):
    """Convert audio to normalized float32."""

    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) == 0:
        raise ValueError("Audio is empty.")

    # Remove DC offset
    audio = audio - np.mean(audio)

    # Normalize amplitude
    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak

    return audio


def get_frames(audio):
    """Split audio into overlapping frames."""

    audio = normalize_audio(audio)

    if len(audio) < FRAME_SIZE:
        audio = np.pad(
            audio,
            (0, FRAME_SIZE - len(audio))
        )

    num_frames = 1 + (
        len(audio) - FRAME_SIZE
    ) // HOP_SIZE

    frames = []

    window = np.hamming(FRAME_SIZE)

    for i in range(num_frames):

        start = i * HOP_SIZE

        frame = audio[
            start:start + FRAME_SIZE
        ]

        if len(frame) < FRAME_SIZE:
            frame = np.pad(
                frame,
                (0, FRAME_SIZE - len(frame))
            )

        frames.append(
            frame * window
        )

    return np.asarray(
        frames,
        dtype=np.float32
    )


def get_filter_bank():
    """Create a Mel-like triangular filter bank."""

    fft_bins = FRAME_SIZE // 2 + 1

    low_freq = 50
    high_freq = SAMPLE_RATE / 2

    filter_points = np.linspace(
        low_freq,
        high_freq,
        NUM_FILTERS + 2
    )

    bins = np.floor(
        (FRAME_SIZE + 1)
        * filter_points
        / SAMPLE_RATE
    ).astype(int)

    bins = np.clip(
        bins,
        0,
        fft_bins - 1
    )

    filter_bank = np.zeros(
        (NUM_FILTERS, fft_bins),
        dtype=np.float32
    )

    for i in range(1, NUM_FILTERS + 1):

        left = bins[i - 1]
        center = bins[i]
        right = bins[i + 1]

        if center <= left:
            center = left + 1

        if right <= center:
            right = center + 1

        right = min(
            right,
            fft_bins - 1
        )

        for j in range(
            left,
            min(center, fft_bins)
        ):

            denominator = max(
                center - left,
                1
            )

            filter_bank[i - 1, j] = (
                (j - left)
                / denominator
            )

        for j in range(
            center,
            right + 1
        ):

            denominator = max(
                right - center,
                1
            )

            filter_bank[i - 1, j] = (
                (right - j)
                / denominator
            )

    return filter_bank


def dct_transform(values):
    """Calculate DCT for MFCC-like cepstral features."""

    n = len(values)

    result = np.zeros(
        NUM_CEPS,
        dtype=np.float32
    )

    for k in range(NUM_CEPS):

        for i in range(n):

            result[k] += (
                values[i]
                * np.cos(
                    np.pi
                    * k
                    * (2 * i + 1)
                    / (2 * n)
                )
            )

    return result


def estimate_pitch(frame):
    """
    Estimate dominant pitch using autocorrelation.
    Returns 0 when pitch is not reliable.
    """

    frame = frame - np.mean(frame)

    energy = np.sum(
        frame ** 2
    )

    if energy < 0.001:
        return 0.0

    correlation = np.correlate(
        frame,
        frame,
        mode="full"
    )

    correlation = correlation[
        len(correlation) // 2:
    ]

    # Human speech pitch range approx. 80–350 Hz
    min_lag = int(
        SAMPLE_RATE / 350
    )

    max_lag = int(
        SAMPLE_RATE / 80
    )

    max_lag = min(
        max_lag,
        len(correlation) - 1
    )

    if max_lag <= min_lag:
        return 0.0

    pitch_region = correlation[
        min_lag:max_lag
    ]

    if len(pitch_region) == 0:
        return 0.0

    best_lag = (
        np.argmax(pitch_region)
        + min_lag
    )

    if best_lag <= 0:
        return 0.0

    return SAMPLE_RATE / best_lag


def extract_speaker_features(audio):
    """
    Extract frame-based speaker characteristics.

    Uses:
    - MFCC-like cepstral features
    - Pitch statistics
    - Energy statistics
    - Zero-crossing statistics
    - Temporal variation
    """

    frames = get_frames(audio)

    filter_bank = get_filter_bank()

    mfcc_features = []
    pitch_values = []
    energy_values = []
    zcr_values = []

    for frame in frames:

        # -----------------------------
        # ENERGY
        # -----------------------------

        energy = np.sqrt(
            np.mean(frame ** 2)
        )

        energy_values.append(
            energy
        )

        # -----------------------------
        # ZERO CROSSING RATE
        # -----------------------------

        zcr = np.mean(
            np.abs(
                np.diff(
                    np.sign(frame)
                )
            )
        ) / 2

        zcr_values.append(
            zcr
        )

        # -----------------------------
        # PITCH
        # -----------------------------

        pitch = estimate_pitch(
            frame
        )

        if pitch > 0:
            pitch_values.append(
                pitch
            )

        # -----------------------------
        # SPECTRUM
        # -----------------------------

        spectrum = np.abs(
            np.fft.rfft(
                frame,
                n=FRAME_SIZE
            )
        )

        power_spectrum = (
            spectrum ** 2
        ) / FRAME_SIZE

        # Filter bank energies
        filtered = np.dot(
            filter_bank,
            power_spectrum
        )

        filtered = np.maximum(
            filtered,
            1e-10
        )

        log_energy = np.log(
            filtered
        )

        # MFCC-like cepstral vector
        mfcc = dct_transform(
            log_energy
        )

        mfcc_features.append(
            mfcc
        )

    mfcc_features = np.asarray(
        mfcc_features,
        dtype=np.float32
    )

    # -----------------------------
    # AGGREGATE MFCC STATISTICS
    # -----------------------------

    mfcc_mean = np.mean(
        mfcc_features,
        axis=0
    )

    mfcc_std = np.std(
        mfcc_features,
        axis=0
    )

    # -----------------------------
    # PITCH STATISTICS
    # -----------------------------

    if len(pitch_values) > 0:

        pitch_mean = np.mean(
            pitch_values
        )

        pitch_std = np.std(
            pitch_values
        )

    else:

        pitch_mean = 0.0
        pitch_std = 0.0

    # -----------------------------
    # ENERGY / ZCR STATISTICS
    # -----------------------------

    energy_values = np.asarray(
        energy_values,
        dtype=np.float32
    )

    zcr_values = np.asarray(
        zcr_values,
        dtype=np.float32
    )

    # -----------------------------
    # FINAL VOICE FINGERPRINT
    # -----------------------------

    features = np.concatenate([

        # Cepstral identity features
        mfcc_mean,
        mfcc_std,

        # Pitch behavior
        np.array([
            pitch_mean,
            pitch_std,
        ], dtype=np.float32),

        # Energy behavior
        np.array([
            np.mean(energy_values),
            np.std(energy_values),
        ], dtype=np.float32),

        # Zero crossing behavior
        np.array([
            np.mean(zcr_values),
            np.std(zcr_values),
        ], dtype=np.float32),

    ])

    # Normalize feature vector
    norm = np.linalg.norm(
        features
    )

    if norm > 0:
        features = (
            features / norm
        )

    return features


def calculate_speaker_similarity(
    registered_audio,
    current_audio,
):
    """
    Compare two voice fingerprints.
    Returns similarity from 0 to 100.
    """

    registered_features = (
        extract_speaker_features(
            registered_audio
        )
    )

    current_features = (
        extract_speaker_features(
            current_audio
        )
    )

    # Cosine similarity
    similarity = np.dot(
        registered_features,
        current_features
    )

    similarity = float(
        np.clip(
            similarity,
            0.0,
            1.0
        )
    )

    return similarity * 100


def get_speaker_consistency_result(
    registered_audio,
    current_audio,
):
    """
    Complete speaker consistency result.
    """

    similarity = (
        calculate_speaker_similarity(
            registered_audio,
            current_audio,
        )
    )

    # Prototype thresholds
    if similarity >= 88:

        status = "CONSISTENT"
        risk = "LOW"

    elif similarity >= 72:

        status = "REVIEW REQUIRED"
        risk = "MEDIUM"

    else:

        status = (
            "POSSIBLE SPEAKER MISMATCH"
        )

        risk = "HIGH"

    return {
        "similarity": round(
            similarity,
            2
        ),
        "status": status,
        "risk": risk,
    }