import json
import random
from pathlib import Path
from typing import List, Dict

class TestCaseGenerator:
    def __init__(self):
        # Chat patterns - conversational, informational queries
        self.chat_patterns = [
            # Greetings
            "hi", "hello", "hey there", "good morning", "good evening",
            "what's up", "how are you", "how's it going", "nice to meet you",
            
            # Questions about assistant
            "what can you do", "who are you", "what's your name",
            "how do you work", "tell me about yourself", "what are your capabilities",
            
            # General questions
            "what's the weather like", "tell me about {topic}",
            "what do you think about {topic}", "can you explain {topic}",
            "how does {topic} work", "why is {topic} important",
            
            # Casual conversation
            "that's interesting", "I see", "tell me more",
            "really?", "wow", "amazing", "cool", "nice",
            
            # Requests for information
            "what is {topic}", "define {topic}", "explain {topic} to me",
            "give me information about {topic}", "I want to learn about {topic}",
            
            # Personal questions
            "do you like {topic}", "what's your opinion on {topic}",
            "how do you feel about {topic}",
            
            # Jokes and entertainment
            "tell me a joke", "make me laugh", "say something funny",
            "tell me a story", "entertain me",
            
            # Advice seeking
            "what should I do about {topic}", "give me advice on {topic}",
            "help me understand {topic}", "I'm confused about {topic}",
        ]
        
        # Command patterns - actionable requests
        self.command_patterns = [
            # App opening
            "open {app}", "launch {app}", "start {app}",
            "can you open {app}", "please open {app}", "open the {app} app",
            
            # Media control
            "play {media}", "pause {media}", "stop {media}",
            "skip this song", "next track", "previous song",
            "play some music", "play my playlist",
            
            # Communication
            "send a message to {person}", "call {person}", "text {person}",
            "email {person}", "message {person} saying {message}",
            "send {message} to {person}",
            
            # Home automation
            "turn on the {device}", "turn off the {device}",
            "dim the {device}", "brighten the {device}",
            "set {device} to {value}", "adjust the {device}",
            
            # Navigation
            "navigate to {location}", "take me to {location}",
            "directions to {location}", "how do I get to {location}",
            "show me the way to {location}",
            
            # Reminders and alarms
            "set a reminder for {task}", "remind me to {task}",
            "set an alarm for {time}", "wake me up at {time}",
            "create a reminder about {task}",
            
            # Search and lookup
            "search for {query}", "look up {query}", "find {query}",
            "google {query}", "search the web for {query}",
            
            # Volume and settings
            "increase volume", "decrease volume", "mute",
            "turn up the volume", "make it louder", "make it quieter",
            "set volume to {value}",
            
            # Calendar and scheduling
            "schedule a meeting with {person}", "add event {event}",
            "create calendar entry for {event}",
            
            # File operations
            "open file {file}", "save this as {file}",
            "delete {file}", "rename {file}",
        ]
        
        # Replacement values
        self.topics = [
            "artificial intelligence", "climate change", "space exploration",
            "quantum computing", "renewable energy", "machine learning",
            "robotics", "cryptocurrency", "virtual reality", "biotechnology"
        ]
        
        self.apps = [
            "WhatsApp", "Spotify", "Chrome", "Gmail", "Calendar",
            "Maps", "YouTube", "Instagram", "Twitter", "Facebook",
            "Slack", "Zoom", "Teams", "Notes", "Camera"
        ]
        
        self.media = [
            "music", "my favorite song", "the radio", "podcast",
            "audiobook", "playlist", "album", "video"
        ]
        
        self.people = [
            "John", "Sarah", "Mom", "Dad", "boss", "colleague",
            "friend", "Alex", "Mike", "Emma", "the team"
        ]
        
        self.messages = [
            "I'll be late", "see you soon", "good morning",
            "check this out", "call me back", "thanks"
        ]
        
        self.devices = [
            "lights", "thermostat", "fan", "AC", "heater",
            "TV", "music system", "bedroom lights", "living room lights"
        ]
        
        self.locations = [
            "home", "office", "nearest gas station", "airport",
            "downtown", "the mall", "central park", "123 Main Street"
        ]
        
        self.tasks = [
            "buy groceries", "call the dentist", "submit report",
            "water the plants", "take medicine", "workout"
        ]
        
        self.times = [
            "7 AM", "8:30 PM", "noon", "midnight", "6 in the morning"
        ]
        
        self.queries = [
            "best restaurants nearby", "latest news", "weather forecast",
            "movie showtimes", "flight tickets", "job openings"
        ]
        
        self.values = [
            "50%", "maximum", "minimum", "medium", "high", "low"
        ]
        
        self.files = [
            "document.pdf", "report.docx", "photo.jpg", "data.csv"
        ]
        
        self.events = [
            "team meeting", "doctor appointment", "lunch with client",
            "conference call", "gym session"
        ]
    
    def _replace_placeholders(self, pattern: str) -> str:
        """Replace placeholders in pattern with actual values."""
        replacements = {
            '{topic}': random.choice(self.topics),
            '{app}': random.choice(self.apps),
            '{media}': random.choice(self.media),
            '{person}': random.choice(self.people),
            '{message}': random.choice(self.messages),
            '{device}': random.choice(self.devices),
            '{location}': random.choice(self.locations),
            '{task}': random.choice(self.tasks),
            '{time}': random.choice(self.times),
            '{query}': random.choice(self.queries),
            '{value}': random.choice(self.values),
            '{file}': random.choice(self.files),
            '{event}': random.choice(self.events),
        }
        
        result = pattern
        for placeholder, value in replacements.items():
            if placeholder in result:
                result = result.replace(placeholder, value)
        
        return result
    
    def _add_variations(self, text: str) -> str:
        """Add natural language variations to make text more realistic."""
        variations = [
            text,  # Original
            text.capitalize(),
            text + "?",
            text + " please",
            "can you " + text,
            "could you " + text,
            "please " + text,
            text + " for me",
            "I need to " + text if random.random() > 0.5 else text,
            text + " now" if random.random() > 0.7 else text,
        ]
        
        return random.choice(variations)
    
    def generate_test_cases(self, num_cases: int = 500) -> List[Dict]:
        """
        Generate test cases with balanced chat/command distribution.
        
        Args:
            num_cases: Total number of test cases to generate
            
        Returns:
            List of test case dictionaries
        """
        test_cases = []
        
        # Split evenly between chat and command
        num_chat = num_cases // 2
        num_command = num_cases - num_chat
        
        # Generate chat cases
        for i in range(num_chat):
            pattern = random.choice(self.chat_patterns)
            text = self._replace_placeholders(pattern)
            text = self._add_variations(text)
            
            test_cases.append({
                "id": f"chat_{i+1:03d}",
                "text": text,
                "expected_intent": "chat"
            })
        
        # Generate command cases
        for i in range(num_command):
            pattern = random.choice(self.command_patterns)
            text = self._replace_placeholders(pattern)
            text = self._add_variations(text)
            
            test_cases.append({
                "id": f"command_{i+1:03d}",
                "text": text,
                "expected_intent": "command"
            })
        
        # Shuffle to mix chat and command cases
        random.shuffle(test_cases)
        
        # Re-assign sequential IDs after shuffle
        for idx, case in enumerate(test_cases):
            case["id"] = f"test_{idx+1:03d}"
        
        return test_cases
    
    def save_test_cases(self, test_cases: List[Dict], output_path: str):
        """Save test cases to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "metadata": {
                "total_cases": len(test_cases),
                "chat_cases": sum(1 for c in test_cases if c["expected_intent"] == "chat"),
                "command_cases": sum(1 for c in test_cases if c["expected_intent"] == "command"),
            },
            "test_cases": test_cases
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Generated {len(test_cases)} test cases")
        print(f"📊 Chat: {data['metadata']['chat_cases']}, Command: {data['metadata']['command_cases']}")
        print(f"💾 Saved to: {output_path}")


def main():
    """Generate test cases and save to file."""
    generator = TestCaseGenerator()
    
    # Generate 500 test cases
    test_cases = generator.generate_test_cases(num_cases=500)
    
    # Save to file
    output_path = "src/tests/data/binary_classifier_test_cases.json"
    generator.save_test_cases(test_cases, output_path)
    
    # Print some examples
    print("\n📝 Sample test cases:")
    print("-" * 70)
    for case in random.sample(test_cases, min(10, len(test_cases))):
        print(f"[{case['expected_intent'].upper()}] {case['text']}")
    print("-" * 70)


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()