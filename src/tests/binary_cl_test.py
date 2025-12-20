import pytest
from core.binary_classifier import predict  # Replace with the actual import path

# Sample test cases
@pytest.mark.parametrize(
    "input_text,expected_label",
    [
        # Commands
        ("can you open settings", 1),
        ("close that file for me", 1),
        ("copy this text", 1),
        ("open whatsapp", 1),
        ("press enter", 1),
        ("Open the File", 1),  # Mixed case
        ("  copy  ", 1),       # Extra spaces

        # Chat
        ("hello, how are you?", 0),
        ("what is your name?", 0),
        ("i had a great day today!", 0),
        ("do you like pizza?", 0),
        ("hi", 0),
        ("12345", 0),            # Numbers treated as chat
        ("", 0),                  # Empty input
        ("   ", 0),               # Spaces only
    ]
)
def test_predict(input_text, expected_label):
    assert predict(input_text) == expected_label

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
