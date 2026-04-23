"""tests/test_lead_capture.py — unit tests for lead capture tool."""

import pytest
from tools.lead_capture import mock_lead_capture, normalize_platform, PLATFORM_MAP


class TestNormalizePlatform:

    @pytest.mark.parametrize("alias,expected", [
        ("youtube",   "YouTube"),
        ("YT",        "YouTube"),
        ("yt",        "YouTube"),
        ("Instagram", "Instagram"),
        ("ig",        "Instagram"),
        ("insta",     "Instagram"),
        ("linkedin",  "LinkedIn"),
        ("li",        "LinkedIn"),
        ("twitter",   "Twitter/X"),
        ("x",         "Twitter/X"),
        ("twitter/x", "Twitter/X"),
        ("tiktok",    "TikTok"),
        ("tt",        "TikTok"),
        ("facebook",  "Facebook"),
        ("fb",        "Facebook"),
    ])
    def test_known_alias(self, alias, expected):
        assert normalize_platform(alias) == expected

    def test_unknown_platform_returns_none(self):
        assert normalize_platform("snapchat") is None

    def test_empty_string_returns_none(self):
        assert normalize_platform("") is None

    def test_case_insensitive(self):
        assert normalize_platform("YOUTUBE") == "YouTube"
        assert normalize_platform("TikTok") == "TikTok"


class TestMockLeadCapture:

    def test_returns_success_status(self):
        result = mock_lead_capture("Alice Smith", "alice@example.com", "YouTube")
        assert result["status"] == "success"

    def test_returns_correct_fields(self):
        result = mock_lead_capture("Bob Jones", "bob@test.io", "LinkedIn")
        assert result["name"]     == "Bob Jones"
        assert result["email"]    == "bob@test.io"
        assert result["platform"] == "LinkedIn"

    def test_lead_id_is_string(self):
        result = mock_lead_capture("Carol", "carol@x.com", "TikTok")
        assert isinstance(result["lead_id"], str)
        assert result["lead_id"].startswith("LEAD-")

    def test_captured_at_is_iso8601(self):
        from datetime import datetime
        result = mock_lead_capture("Dave", "dave@y.com", "Instagram")
        # Should parse without error
        datetime.fromisoformat(result["captured_at"])

    def test_source_field(self):
        result = mock_lead_capture("Eve", "eve@z.com", "Twitter/X")
        assert result["source"] == "AutoStream-Agent"
