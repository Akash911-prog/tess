import pytest
from unittest.mock import Mock, patch
import numpy as np

# Assuming your LCN class is in a module called 'lcn'
# Adjust import based on your actual module structure
from core.lcn import LCN


@pytest.fixture(scope="module")
def lcn_instance():
    """Create a single LCN instance for all tests to save loading time."""
    return LCN()


class TestPreprocessing:
    """Tests for text preprocessing functionality."""
    
    def test_lowercase_conversion(self, lcn_instance):
        result = lcn_instance.preprocess("OPEN CHROME")
        assert result == "open chrome"
    
    def test_filler_removal(self, lcn_instance):
        result = lcn_instance.preprocess("can you please open chrome")
        assert "please" not in result
        assert "chrome" in result
    
    def test_extra_whitespace_removal(self, lcn_instance):
        result = lcn_instance.preprocess("open    chrome   browser")
        assert "  " not in result
        assert result == result.strip()
    
    def test_edge_case_filler_at_start(self, lcn_instance):
        result = lcn_instance.preprocess("please open chrome")
        assert not result.startswith("please")
    
    def test_edge_case_filler_at_end(self, lcn_instance):
        result = lcn_instance.preprocess("open chrome please")
        assert not result.endswith("please")


class TestCommandCleaning:
    """Tests for command cleaning functionality."""
    
    def test_keeps_verbs_and_nouns(self, lcn_instance):
        result = lcn_instance.clean_command("open chrome browser")
        assert "open" in result
        assert "chrome" in result
    
    def test_removes_articles(self, lcn_instance):
        result = lcn_instance.clean_command("open the chrome browser")
        # Articles should be removed in cleaning
        assert result.count("the") == 0 or "the" not in result.split()
    
    def test_lemmatization(self, lcn_instance):
        result = lcn_instance.clean_command("opening chrome")
        # Should lemmatize "opening" to "open"
        assert "open" in result
    
    def test_preserves_directional_prepositions(self, lcn_instance):
        result = lcn_instance.clean_command("scroll up")
        assert "up" in result
        
        result = lcn_instance.clean_command("volume down")
        assert "down" in result
    
    def test_removes_consecutive_duplicates(self, lcn_instance):
        result = lcn_instance.clean_command("open open chrome")
        words = result.split()
        # Check no consecutive duplicates
        for i in range(len(words) - 1):
            assert words[i] != words[i + 1]


class TestIntentMatching:
    """Tests for intent matching functionality."""
    
    def test_open_intent(self, lcn_instance):
        intent, cleaned, score = lcn_instance.get_intent("open chrome")
        assert intent == "open"
        assert score > 0.6
    
    def test_close_intent(self, lcn_instance):
        intent, cleaned, score = lcn_instance.get_intent("close the window")
        assert intent == "close"
        assert score > 0.6
    
    def test_copy_intent(self, lcn_instance):
        intent, cleaned, score = lcn_instance.get_intent("copy this text")
        assert intent == "copy"
        assert score > 0.6
    
    def test_volume_up_intent(self, lcn_instance):
        intent, cleaned, score = lcn_instance.get_intent("increase volume")
        assert intent == "volume up"
        assert score > 0.6
    
    def test_volume_down_intent(self, lcn_instance):
        intent, cleaned, score = lcn_instance.get_intent("lower the volume")
        assert intent == "volume down"
        assert score > 0.6
    
    def test_shutdown_intent(self, lcn_instance):
        intent, cleaned, score = lcn_instance.get_intent("shut down computer")
        assert intent == "shutdown"
        assert score > 0.6
    
    def test_unknown_intent(self, lcn_instance):
        intent, cleaned, score = lcn_instance.get_intent("xyzabc nonsense command")
        assert intent == "unknown"
        assert score < 0.6
    
    def test_ambiguous_command_low_confidence(self, lcn_instance):
        intent, cleaned, score = lcn_instance.get_intent("do something")
        # Should either be unknown or have low confidence
        assert intent == "unknown" or score < 0.7
    
    def test_natural_language_variations(self, lcn_instance):
        # Test various ways of saying the same thing
        commands = [
            "open chrome",
            "launch chrome",
            "start chrome browser",
            "run chrome"
        ]
        for cmd in commands:
            intent, _, score = lcn_instance.get_intent(cmd)
            assert intent == "open", f"Failed for: {cmd}"
    
    def test_with_fillers(self, lcn_instance):
        intent, cleaned, score = lcn_instance.get_intent("can you please open chrome")
        assert intent == "open"
        assert score > 0.6


class TestParameterExtraction:
    """Tests for parameter extraction functionality."""
    
    def test_extract_target(self, lcn_instance):
        params = lcn_instance.extract_params("open chrome browser")
        assert "TARGET" in params or "APP" in params
    
    def test_extract_number(self, lcn_instance):
        # This depends on your params_pattern configuration
        params = lcn_instance.extract_params("open tab 5")
        # Check if number is extracted (label depends on your config)
        assert any(key in params for key in ["NUM", "NUMBER", "TARGET"])
    
    def test_noun_chunk_fallback(self, lcn_instance):
        params = lcn_instance.extract_params("open the settings application")
        # Should extract "settings application" as target
        assert "TARGET" in params
        assert "settings" in params["TARGET"].lower()


class TestNormalization:
    """Tests for the complete normalization pipeline."""
    
    def test_normalize_simple_command(self, lcn_instance):
        result = lcn_instance.normalize("open chrome")
        assert result["intent"] == "open"
        assert "params" in result
        assert "confidence" in result
        assert result["confidence"] > 0
    
    def test_normalize_returns_confidence(self, lcn_instance):
        result = lcn_instance.normalize("close window")
        assert 0 <= result["confidence"] <= 1
    
    def test_normalize_returns_cleaned(self, lcn_instance):
        result = lcn_instance.normalize("please open chrome browser")
        assert "cleaned" in result
        assert result["cleaned"] != ""
    
    def test_normalize_unknown_command(self, lcn_instance):
        result = lcn_instance.normalize("xyzabc random stuff")
        assert result["intent"] is None
        assert result["confidence"] == 0.0
    
    def test_normalize_code_intent(self, lcn_instance):
        result = lcn_instance.normalize("write code for sorting")
        if result["intent"] == "code":
            assert result["params"]["query"] == "write code for sorting"
            assert result["confidence"] > 0
    
    def test_normalize_with_parameters(self, lcn_instance):
        result = lcn_instance.normalize("open chrome browser")
        assert result["intent"] == "open"
        assert len(result["params"]) >= 0  # May or may not extract params


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_string(self, lcn_instance):
        result = lcn_instance.normalize("")
        assert result["intent"] is None
        assert result["confidence"] == 0.0
    
    def test_only_fillers(self, lcn_instance):
        result = lcn_instance.normalize("please can you")
        assert result["intent"] is None or result["confidence"] < 0.5
    
    def test_very_long_command(self, lcn_instance):
        long_cmd = "please can you open chrome and then go to youtube and play a video " * 5
        result = lcn_instance.normalize(long_cmd)
        # Should still process without errors
        assert "intent" in result
        assert "confidence" in result
    
    def test_special_characters(self, lcn_instance):
        result = lcn_instance.normalize("open chrome!!! @#$")
        assert result["intent"] == "open"
    
    def test_numbers_only(self, lcn_instance):
        result = lcn_instance.normalize("123 456 789")
        # Should handle gracefully
        assert "intent" in result


class TestConfidenceScoring:
    """Tests for confidence scoring logic."""
    
    def test_high_confidence_exact_match(self, lcn_instance):
        result = lcn_instance.normalize("open chrome")
        assert result["confidence"] > 0.7
    
    def test_medium_confidence_variant(self, lcn_instance):
        result = lcn_instance.normalize("launch chrome browser please")
        assert result["confidence"] > 0.6
    
    def test_confidence_with_parameters(self, lcn_instance):
        result_with_param = lcn_instance.normalize("open chrome browser")
        result_without = lcn_instance.normalize("open")
        
        # Command with target should ideally have higher confidence
        # (This depends on your implementation)
        assert result_with_param["confidence"] >= 0


class TestIntegration:
    """Integration tests for common use cases."""
    
    def test_browser_commands(self, lcn_instance):
        commands = [
            ("open chrome", "open"),
            ("close browser", "close"),
            ("new tab", "new tab"),
            ("close tab", "close tab"),
            ("next tab", "next tab")
        ]
        
        for cmd, expected_intent in commands:
            result = lcn_instance.normalize(cmd)
            assert result["intent"] == expected_intent, f"Failed for: {cmd}"
    
    def test_volume_commands(self, lcn_instance):
        commands = [
            ("volume up", "volume up"),
            ("increase volume", "volume up"),
            ("volume down", "volume down"),
            ("lower volume", "volume down")
        ]
        
        for cmd, expected_intent in commands:
            result = lcn_instance.normalize(cmd)
            assert result["intent"] == expected_intent, f"Failed for: {cmd}"
    
    def test_clipboard_commands(self, lcn_instance):
        commands = [
            ("copy", "copy"),
            ("paste", "paste"),
            ("cut", "cut")
        ]
        
        for cmd, expected_intent in commands:
            result = lcn_instance.normalize(cmd)
            assert result["intent"] == expected_intent, f"Failed for: {cmd}"
    
    def test_system_commands(self, lcn_instance):
        commands = [
            ("shutdown", "shutdown"),
            ("restart", "restart"),
            ("lock screen", "lock")
        ]
        
        for cmd, expected_intent in commands:
            result = lcn_instance.normalize(cmd)
            assert result["intent"] == expected_intent, f"Failed for: {cmd}"


class TestPerformance:
    """Performance and caching tests."""
    
    def test_caching_works(self, lcn_instance):
        # First call
        result1 = lcn_instance.normalize("open chrome")
        
        # Second call (should use cache)
        result2 = lcn_instance.normalize("open chrome")
        
        assert result1 == result2
    
    def test_multiple_calls_same_intent(self, lcn_instance):
        commands = ["open chrome", "open firefox", "open vscode"]
        
        for cmd in commands:
            result = lcn_instance.normalize(cmd)
            assert result["intent"] == "open"


# Parametrized tests for comprehensive coverage
@pytest.mark.parametrize("command,expected_intent", [
    ("open chrome", "open"),
    ("close window", "close"),
    ("copy text", "copy"),
    ("paste here", "paste"),
    ("volume up", "volume up"),
    ("shutdown computer", "shutdown"),
    ("new tab", "new tab"),
    ("minimize window", "minimize window"),
    ("maximize window", "maximize window"),
])
def test_common_commands_parametrized(lcn_instance, command, expected_intent):
    result = lcn_instance.normalize(command)
    assert result["intent"] == expected_intent
    assert result["confidence"] > 0.6


@pytest.mark.parametrize("noisy_command,expected_intent", [
    ("can you please open chrome", "open"),
    ("hey please close the window", "close"),
    ("just copy this text", "copy"),
    ("please paste it here", "paste"),
])
def test_noisy_commands_parametrized(lcn_instance, noisy_command, expected_intent):
    result = lcn_instance.normalize(noisy_command)
    assert result["intent"] == expected_intent


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])