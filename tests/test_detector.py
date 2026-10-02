from modules.ai_detector import get_detector


print("================================")
print(" VoiceSentinel AI Test")
print("================================")

detector = get_detector()

info = detector.check_environment()

print()
print("PyTorch version :", info["torch_version"])
print("Torchaudio      :", info["torchaudio_version"])
print("Device          :", info["device"])
print("Detector status :", detector.get_status())

print()
print("Environment test completed!")