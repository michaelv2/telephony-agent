"""Tests for turn detection heuristics.

Covers the fast-path logic that determines whether a transcript is a complete
conversational turn (fire response immediately) or an incomplete fragment
(hold and wait for more speech).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Mock heavy dependencies that server.py imports at module level
from unittest.mock import MagicMock

sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("faster_whisper", MagicMock())
sys.modules.setdefault("twilio", MagicMock())
sys.modules.setdefault("twilio.request_validator", MagicMock())
sys.modules.setdefault("twilio.rest", MagicMock())

# Provide required env vars before importing server
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("TWILIO_ACCOUNT_SID", "test")
os.environ.setdefault("TWILIO_AUTH_TOKEN", "test")
os.environ.setdefault("TWILIO_API_KEY", "test")
os.environ.setdefault("TWILIO_API_SECRET", "test")
os.environ.setdefault("TWILIO_PHONE", "+10000000000")

from server import _is_obviously_complete, _is_complete_turn

_is_complete_turn_heuristic = _is_complete_turn


# ============================================================
# Tests for _is_obviously_complete (original fast-path, preserved)
# ============================================================

class TestObviouslyComplete:
    def test_question_mark_in_tail(self):
        assert _is_obviously_complete("Is that OK?")

    def test_question_mark_mid_sentence_in_tail(self):
        assert _is_obviously_complete("I was wondering, can you help?")

    def test_no_question_no_pattern(self):
        assert not _is_obviously_complete("I would like to schedule an appointment.")

    def test_pattern_what_do_you_think(self):
        assert _is_obviously_complete("That's my idea, what do you think")

    def test_pattern_go_ahead(self):
        assert _is_obviously_complete("Sure, go ahead")

    def test_pattern_right_question(self):
        assert _is_obviously_complete("That's correct, right?")

    def test_empty_string(self):
        assert not _is_obviously_complete("")

    def test_short_fragment(self):
        assert not _is_obviously_complete("Well")

    def test_question_mark_too_far_back(self):
        # "?" is beyond the 60-char tail window
        long_tail = "?" + " " + "a" * 70
        assert not _is_obviously_complete(long_tail)


# ============================================================
# Tests for _is_complete_turn_heuristic (new logic)
# ============================================================

class TestCompleteHeuristic:
    """Real utterances from transcripts and realistic conversation scenarios."""

    # --- Should be COMPLETE ---

    def test_declarative_sentence_with_period(self):
        assert _is_complete_turn_heuristic(
            "I was just calling to see if you are still available."
        )

    def test_appointment_request(self):
        assert _is_complete_turn_heuristic(
            "I would like to schedule an appointment for my daughter."
        )

    def test_question_with_mark(self):
        assert _is_complete_turn_heuristic(
            "Is that OK?"
        )

    def test_exclamation(self):
        assert _is_complete_turn_heuristic(
            "That sounds amazing!"
        )

    def test_long_declarative(self):
        assert _is_complete_turn_heuristic(
            "Yeah, we actually, just a couple weeks ago, got back from "
            "a trip to Vietnam and Japan."
        )

    def test_short_complete_response(self):
        assert _is_complete_turn_heuristic(
            "No problem at all."
        )

    def test_farewell(self):
        assert _is_complete_turn_heuristic(
            "Actually, I know what I have to go. Can I call you back?"
        )

    def test_multi_sentence(self):
        assert _is_complete_turn_heuristic(
            "I'm sorry, I didn't catch all of that. Can you repeat it?"
        )

    def test_pattern_based_complete(self):
        assert _is_complete_turn_heuristic(
            "That's my proposal, what do you think"
        )

    def test_casual_agreement(self):
        assert _is_complete_turn_heuristic(
            "Yeah that works for me."
        )

    # --- Should be INCOMPLETE ---

    def test_trailing_and(self):
        assert not _is_complete_turn_heuristic(
            "I wanted to schedule something and"
        )

    def test_trailing_but(self):
        assert not _is_complete_turn_heuristic(
            "That sounds good but"
        )

    def test_trailing_because(self):
        assert not _is_complete_turn_heuristic(
            "I'm calling because"
        )

    def test_trailing_to(self):
        assert not _is_complete_turn_heuristic(
            "I was hoping to"
        )

    def test_trailing_the(self):
        assert not _is_complete_turn_heuristic(
            "Can you check the"
        )

    def test_bare_fragment_no_punctuation(self):
        assert not _is_complete_turn_heuristic(
            "Well actually"
        )

    def test_single_word(self):
        assert not _is_complete_turn_heuristic("Um")

    def test_empty(self):
        assert not _is_complete_turn_heuristic("")

    def test_trailing_conjunction_with(self):
        assert not _is_complete_turn_heuristic(
            "I need to talk to someone with"
        )

    def test_very_short_with_period(self):
        # "OK." is only 1 word — too short to be confident
        assert not _is_complete_turn_heuristic("OK.")

    def test_two_words_with_period(self):
        assert not _is_complete_turn_heuristic("Sure thing.")

    def test_three_words_with_period_is_enough(self):
        assert _is_complete_turn_heuristic("Got it now.")

    def test_four_words_with_period(self):
        assert _is_complete_turn_heuristic("That works for me.")

    # --- Edge cases ---

    def test_trailing_i_m(self):
        assert not _is_complete_turn_heuristic("The thing is I'm")

    def test_trailing_it_s(self):
        assert not _is_complete_turn_heuristic("What I meant was it's")

    def test_period_then_trailing_conjunction(self):
        # "I agree. But" — has a period mid-text, but trails with "but"
        assert not _is_complete_turn_heuristic("I agree. But")

    def test_real_transcript_vietnam_midthought(self):
        # Trails with " is" — user continued speaking 20s later (12:28:46).
        # Holding is correct here.
        assert not _is_complete_turn_heuristic(
            "Yeah, they were really different. Japan is very modern. "
            "There are a lot of deep cultural traditions, whereas Senui is"
        )

    def test_real_transcript_vietnam_continued(self):
        # The full utterance — clearly complete
        assert _is_complete_turn_heuristic(
            "a typical growth market where there's still a lot of extreme "
            "income inequality and in a way the mindset is a bit more "
            "Western focus given how much tourism is a focus and food culture "
            "is also probably very different but yeah it was kind of surprised "
            "how different they are."
        )
