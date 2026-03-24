# Contributing

## Layout

Application code lives in **`app/`**. The dashboard is **`static/dashboard.html`**. See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for data flow.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_pose_model.py
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Optional: `LOG_LEVEL=DEBUG` for verbose pose pipeline logs.

## Tests

```bash
pytest tests/ -q
```

Golden regression runs only when `tests/fixtures/golden_shot.mp4` and `golden_expected.json` are present (see `docs/GOLDEN_VIDEO_GUIDE.md`).

## Style

- Prefer **`logging`** over **`print`** in application code.
- Keep API response fields backward-compatible when possible.
