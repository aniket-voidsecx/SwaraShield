from modules.risk_engine import calculate_voice_risk


print("================================")
print(" VoiceSentinel Risk Engine Test")
print("================================")


# Simulated AASIST result
spoof_probability = 99.92


result = calculate_voice_risk(
    spoof_probability
)


print()
print(
    "Spoof Probability :",
    spoof_probability,
    "%"
)

print(
    "Risk Score        :",
    result["risk_score"]
)

print(
    "Risk Level        :",
    result["risk_level"]
)

print(
    "Recommendation    :",
    result["recommendation"]
)

print()
print("Risk Engine working successfully.")