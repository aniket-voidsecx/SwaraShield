# Source

Recommended responsibilities:

- `interceptor/` — real-time media subscription/interception logic
- `audio/` — audio conversion, resampling, chunking, normalization
- `models/` — model loading and inference adapters
- `risk_engine/` — score fusion and status decisions
- `utils/` — shared helpers and configuration

Keep modules small and testable.
