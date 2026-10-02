from speaker_verification import (
    verify_speaker,
    get_speaker_risk_score,
)


print("=" * 60)
print(" VoiceSentinel - Speaker Verification Engine Test")
print("=" * 60)

registered_voice = "data/sample_audio/genuine_voice.wav"
current_voice = "data/sample_audio/test_pcm.wav"

print("\nRegistered voice:", registered_voice)
print("Current voice   :", current_voice)

print("\nLoading model and comparing voices...")

result = verify_speaker(
    registered_voice,
    current_voice,
)

risk_score = get_speaker_risk_score(
    registered_voice,
    current_voice,
)

print("\n" + "=" * 60)
print(" SPEAKER CONSISTENCY RESULT")
print("=" * 60)

print(f"Similarity Score : {result['similarity']}%")
print(f"Raw Score        : {result['raw_score']}")
print(f"Status           : {result['status']}")
print(f"Risk Level       : {result['risk']}")
print(f"Speaker Risk     : {risk_score}/100")

print("=" * 60)
print("\nTest completed successfully.")
