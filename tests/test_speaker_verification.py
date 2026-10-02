from speechbrain.inference.speaker import SpeakerRecognition


print("=" * 60)
print(" VoiceSentinel - Pretrained Speaker Verification Test")
print("=" * 60)

print("\nLoading pretrained speaker verification model...")

verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/speaker_ecapa"
)

print("Model loaded successfully.")

registered_voice = "data/sample_audio/genuine_voice.wav"
current_voice = "data/sample_audio/genuine_voice.wav"

print("\nRegistered voice:", registered_voice)
print("Current voice   :", current_voice)

print("\nRunning speaker verification...")

score, prediction = verification.verify_files(
    registered_voice,
    current_voice
)

print("\n" + "=" * 60)
print(" SPEAKER VERIFICATION RESULT")
print("=" * 60)

print("Raw similarity score :", float(score))
print("Same speaker         :", bool(prediction))

if bool(prediction):
    print("Status               : CONSISTENT")
else:
    print("Status               : POSSIBLE SPEAKER MISMATCH")

print("=" * 60)
print("\nTest completed.")