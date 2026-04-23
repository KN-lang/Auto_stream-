"""
tools/lead_capture.py
=====================
Lead capture tool — writes to Google Sheets (production) or mock (dev/test).

Configuration
-------------
Set these in .env to enable Google Sheets:
  GOOGLE_SERVICE_ACCOUNT_JSON = '{"type":"service_account",...}'   (minified JSON)
  GOOGLE_SHEET_ID             = 'your_google_sheet_id_here'

If either is missing, falls back to mock_lead_capture automatically.

Sheet columns (row appended on each capture):
  A: Lead ID | B: Timestamp | C: Name | D: Email | E: Platform | F: Source
"""

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Platform normalisation map ─────────────────────────────────────────────────
PLATFORM_MAP: dict[str, str] = {
    "youtube":   "YouTube",
    "yt":        "YouTube",
    "instagram": "Instagram",
    "ig":        "Instagram",
    "insta":     "Instagram",
    "linkedin":  "LinkedIn",
    "li":        "LinkedIn",
    "twitter":   "Twitter/X",
    "twitter/x": "Twitter/X",
    "x":         "Twitter/X",
    "tiktok":    "TikTok",
    "tik tok":   "TikTok",
    "tt":        "TikTok",
    "facebook":  "Facebook",
    "fb":        "Facebook",
    "twitch":    "Twitch",
}


def normalize_platform(raw: str) -> str | None:
    """Map a user-supplied platform string to a canonical name."""
    return PLATFORM_MAP.get(raw.strip().lower())


# ── Google Sheets integration ──────────────────────────────────────────────────

def _sheets_available() -> bool:
    return bool(
        os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        and os.getenv("GOOGLE_SHEET_ID")
    )


def _append_to_sheet(lead_id: str, name: str, email: str,
                     platform: str, captured_at: str) -> None:
    """Append a lead row to the configured Google Sheet."""
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    service_account_info = json.loads(raw_json)
    creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
    client = gspread.authorize(creds)

    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    spreadsheet = client.open_by_key(sheet_id)
    worksheet = spreadsheet.sheet1   # writes to the first tab

    worksheet.append_row(
        [lead_id, captured_at, name, email, platform, "AutoStream-Agent"],
        value_input_option="USER_ENTERED",
    )
    logger.info("[sheets] Lead appended → Sheet ID %s  row: %s", sheet_id, lead_id)


# ── Mock fallback ──────────────────────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Simulates persisting a qualified lead (used when Google Sheets is not configured).
    Returns the same schema as the real integration so callers are unaffected.
    """
    lead_id = f"LEAD-{abs(hash(email)) % 100_000:05d}"
    captured_at = datetime.now(timezone.utc).isoformat()

    payload = {
        "status":      "success",
        "lead_id":     lead_id,
        "name":        name,
        "email":       email,
        "platform":    platform,
        "captured_at": captured_at,
        "source":      "AutoStream-Agent",
        "backend":     "mock",
    }

    logger.info(
        "[mock_lead_capture] id=%s name=%s email=%s platform=%s",
        lead_id, name, email, platform,
    )
    _print_receipt(payload)
    return payload


# ── Public entry point ─────────────────────────────────────────────────────────

def capture_lead(name: str, email: str, platform: str) -> dict:
    """
    Primary lead capture function used by the agent graph.

    Tries Google Sheets first; falls back to mock if credentials
    are not configured. Always returns a consistent receipt dict.
    """
    lead_id = f"LEAD-{abs(hash(email)) % 100_000:05d}"
    captured_at = datetime.now(timezone.utc).isoformat()

    payload = {
        "status":      "success",
        "lead_id":     lead_id,
        "name":        name,
        "email":       email,
        "platform":    platform,
        "captured_at": captured_at,
        "source":      "AutoStream-Agent",
    }

    if _sheets_available():
        try:
            _append_to_sheet(lead_id, name, email, platform, captured_at)
            payload["backend"] = "google_sheets"
            logger.info("[capture_lead] ✅ Written to Google Sheets — %s", lead_id)
        except Exception as exc:
            logger.warning(
                "[capture_lead] Google Sheets failed (%s) — falling back to mock", exc
            )
            payload["backend"] = "mock_fallback"
    else:
        payload["backend"] = "mock"
        logger.info("[capture_lead] Sheets not configured — using mock backend")

    _print_receipt(payload)
    return payload


def _print_receipt(payload: dict) -> None:
    backend_label = {
        "google_sheets": "✅ Google Sheets",
        "mock":          "🔶 Mock (dev mode)",
        "mock_fallback": "⚠️  Mock (Sheets failed)",
    }.get(payload.get("backend", "mock"), "Mock")

    print(f"\n{'='*55}")
    print(f"  LEAD CAPTURED  [{backend_label}]")
    print(f"{'='*55}")
    print(f"  Lead ID   : {payload['lead_id']}")
    print(f"  Name      : {payload['name']}")
    print(f"  Email     : {payload['email']}")
    print(f"  Platform  : {payload['platform']}")
    print(f"  Timestamp : {payload['captured_at']}")
    print(f"{'='*55}\n")
