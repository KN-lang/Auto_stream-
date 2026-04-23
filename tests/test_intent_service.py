"""tests/test_intent_service.py — unit tests for IntentService."""

import pytest
from unittest.mock import MagicMock, patch
from services.intent_service import IntentService, IntentClassification


def _make_result(intent, confidence=0.9, reasoning="test"):
    return IntentClassification(intent=intent, confidence=confidence, reasoning=reasoning)


@pytest.fixture
def mock_service():
    """
    Return an IntentService whose classify() loop is short-circuited:
    we patch make_llm_for inside intent_service so no real API is called.
    """
    svc = IntentService.__new__(IntentService)
    # Build a mock extractor that returns a preset result
    mock_extractor = MagicMock()
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_extractor

    with patch("services.llm_factory.make_llm_for", return_value=mock_llm):
        svc = IntentService()
        svc._mock_extractor = mock_extractor
    return svc


def _classify_with(mock_service, result, message="test"):
    """Drive a classify() call by pre-setting the mock extractor return value."""
    mock_service._mock_extractor.invoke.return_value = result
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_service._mock_extractor
    with patch("services.llm_factory.make_llm_for", return_value=mock_llm):
        return mock_service.classify(message)


class TestIntentClassification:

    def test_greeting_hello(self, mock_service):
        r = _classify_with(mock_service, _make_result("greeting"), "Hey there!")
        assert r.intent == "greeting"

    def test_product_inquiry_price(self, mock_service):
        r = _classify_with(mock_service, _make_result("product_inquiry"), "How much is Pro?")
        assert r.intent == "product_inquiry"

    def test_product_inquiry_refund(self, mock_service):
        r = _classify_with(mock_service, _make_result("product_inquiry"), "What is your refund policy?")
        assert r.intent == "product_inquiry"

    def test_high_intent_signup(self, mock_service):
        r = _classify_with(mock_service, _make_result("high_intent_lead"), "I want to sign up!")
        assert r.intent == "high_intent_lead"

    def test_high_intent_lets_get_started(self, mock_service):
        r = _classify_with(mock_service, _make_result("high_intent_lead"), "Let's get started.")
        assert r.intent == "high_intent_lead"

    def test_curiosity_stays_product_inquiry(self, mock_service):
        r = _classify_with(mock_service, _make_result("product_inquiry"), "How much does Pro cost?")
        assert r.intent == "product_inquiry"

    def test_confidence_range(self, mock_service):
        r = _classify_with(mock_service, _make_result("greeting", confidence=0.95), "Hi!")
        assert 0.0 <= r.confidence <= 1.0

    def test_reasoning_is_non_empty(self, mock_service):
        r = _classify_with(mock_service, _make_result("greeting", reasoning="It is a greeting."), "Hi!")
        assert len(r.reasoning) > 0
