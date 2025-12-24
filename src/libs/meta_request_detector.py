from typing import Optional
import logging

logger = logging.getLogger(__name__)

class MetaRequestDetector:
    def __init__(self, nlp):
        """Initialize spaCy model for meta-request detection"""
        self.nlp = nlp
        
        # Expanded meta verbs and patterns
        self.meta_verbs = {
            "write", "generate", "create", "make", "build", "develop",
            "explain", "describe", "clarify", "elaborate", "detail",
            "summarize", "outline", "overview", "recap",
            "debug", "fix", "solve", "troubleshoot", "diagnose",
            "analyze", "examine", "evaluate", "assess", "review",
            "design", "plan", "architect", "structure",
            "teach", "show", "demonstrate", "illustrate", "guide",
            "help", "assist", "support",
            "refactor", "optimize", "improve", "enhance",
            "compare", "contrast", "differentiate",
            "translate", "convert", "transform",
            "implement", "code", "program",
            "test", "validate", "verify", "check"
        }
        
        self.meta_phrases = {
            "help me", "show me", "walk me through", "guide me",
            "can you", "could you", "would you", "please",
            "i need", "i want", "i'd like",
            "how do i", "how can i", "how to"
        }
    
    def is_meta_request(self, text: str) -> bool:
        """
        Detect if text is a meta-request (asking for creation, explanation, help)
        Uses both spaCy linguistic analysis and rule-based patterns
        """
        text_lower = text.lower().strip()
        
        # Rule 1: Check for meta phrases (fast check)
        for phrase in self.meta_phrases:
            if phrase in text_lower:
                logger.debug(f"Meta-request detected: phrase '{phrase}'")
                return True
        
        # Rule 2: Simple keyword check (fallback for when spaCy unavailable)
        for verb in self.meta_verbs:
            if verb in text_lower:
                logger.debug(f"Meta-request detected: keyword '{verb}'")
                return True
        
        # spaCy-based detection (more sophisticated)
        if self.nlp is not None:
            doc = self.nlp(text)
            
            # Rule 3: Check for imperative mood (commands)
            # Imperative sentences often start with a base verb
            if len(doc) > 0:
                first_token = doc[0]
                # Check if first token is a verb in base form (VB) or modal (MD)
                if first_token.pos_ == "VERB" and first_token.tag_ == "VB":
                    if first_token.lemma_ in self.meta_verbs:
                        logger.debug(f"Meta-request detected: imperative '{first_token.lemma_}' (spaCy)")
                        return True
            
            # Rule 4: Check main verbs in the sentence
            for token in doc:
                # Look for root verbs (main verb of sentence)
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    if token.lemma_ in self.meta_verbs:
                        logger.debug(f"Meta-request detected: root verb '{token.lemma_}' (spaCy)")
                        return True
                
                # Look for verbs that are direct objects of auxiliary/modal constructions
                # E.g., "Can you explain..." - "explain" is xcomp of "can"
                if token.dep_ == "xcomp" and token.pos_ == "VERB":
                    if token.lemma_ in self.meta_verbs:
                        logger.debug(f"Meta-request detected: complement verb '{token.lemma_}' (spaCy)")
                        return True
            
            # Rule 5: Check for requests with modal verbs + meta verbs
            # E.g., "Could you help me", "Would you write"
            has_modal = False
            has_meta_verb = False
            
            for token in doc:
                if token.tag_ == "MD":  # Modal verb
                    has_modal = True
                if token.pos_ == "VERB" and token.lemma_ in self.meta_verbs:
                    has_meta_verb = True
            
            if has_modal and has_meta_verb:
                logger.debug(f"Meta-request detected: modal + meta verb (spaCy)")
                return True
            
            # Rule 6: Check for "need/want + to + meta_verb" pattern
            # E.g., "I need to create", "I want to understand"
            for token in doc:
                if token.lemma_ in ["need", "want", "like"] and token.pos_ == "VERB":
                    # Check if followed by infinitive marker "to" and a meta verb
                    for child in token.children:
                        if child.dep_ == "xcomp" and child.lemma_ in self.meta_verbs:
                            logger.debug(f"Meta-request detected: need/want pattern (spaCy)")
                            return True
            
            # Rule 7: Check for gerunds/present participles as objects
            # E.g., "I'm interested in learning", "looking for help with creating"
            for token in doc:
                if token.tag_ == "VBG" and token.lemma_ in self.meta_verbs:
                    if token.dep_ in ["pobj", "dobj", "xcomp"]:
                        logger.debug(f"Meta-request detected: gerund object '{token.lemma_}' (spaCy)")
                        return True
        
        return False

# Standalone function (for backward compatibility)
def is_meta_request(text: str, detector: Optional[MetaRequestDetector] = None) -> bool:
    """
    Convenience function for meta-request detection
    
    Args:
        text: Input text to analyze
        detector: Optional MetaRequestDetector instance (creates one if not provided)
    
    Returns:
        bool: True if text is a meta-request
    """
    if detector is None:
        detector = MetaRequestDetector()
    return detector.is_meta_request(text)


# Example usage:
if __name__ == "__main__":
    detector = MetaRequestDetector()
    
    test_cases = [
        "Write me a function to sort arrays",
        "Can you explain how recursion works?",
        "I need help debugging this code",
        "Generate a report for Q4 sales",
        "Show me how to implement a binary tree",
        "The weather is nice today",  # Not a meta-request
        "Please design a database schema",
        "Looking for help with creating a REST API"
    ]
    
    for test in test_cases:
        result = detector.is_meta_request(test)
        print(f"'{test}' -> {result}")