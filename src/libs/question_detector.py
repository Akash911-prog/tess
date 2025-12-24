import logging

logger = logging.getLogger(__name__)

class QuestionDetector:
    def __init__(self, nlp):
        """Initialize spaCy model for question detection"""
        self.nlp = nlp
    
    def _is_question_pattern(self, text: str) -> bool:
        """Detect if text is a question using both spaCy and rule-based signals"""
        text_lower = text.lower().strip()
        
        # Rule 1: Question mark
        if text_lower.endswith("?"):
            logger.debug("Question detected: ends with '?'")
            return True
        
        # Rule 2: Question starters
        question_starters = [
            "what is", "what are", "what's", "whats",
            "who is", "who are", "who's", "whos",
            "when is", "when are", "when was", "when's",
            "where is", "where are", "where's",
            "how do", "how does", "how can", "how to", "how about",
            "why is", "why are", "why does", "why don't", "why not",
            "which is", "which are", "which one",
            "tell me", "show me", "give me", "explain",
            "find out", "look up", "search for",
            "do you", "did you", "have you", "has anyone",
            "can you", "could you", "would you", "will you",
            "is there", "are there", "was there", "were there"
        ]
        
        for starter in question_starters:
            if text_lower.startswith(starter):
                logger.debug(f"Question detected: starts with '{starter}'")
                return True
        
        # Rule 3: Question words in first 2 words
        words = text_lower.split()[:2]
        question_words = {"what", "who", "when", "where", "why", "how", "is", "are", "can", "could", "would", "will", "should", "did", "does", "do"}
        if any(word in question_words for word in words):
            logger.debug(f"Question detected: question word in first 2 words")
            return True
        
        # spaCy-based detection (if available)
        if self.nlp is not None:
            doc = self.nlp(text)
            
            # Rule 4: Check dependency patterns (auxiliary verb inversion)
            # E.g., "Is this correct?" - auxiliary "is" comes before subject
            for token in doc:
                if token.dep_ == "aux" and token.head.pos_ == "VERB":
                    # Check if auxiliary comes before its head (verb inversion)
                    if token.i < token.head.i:
                        logger.debug(f"Question detected: auxiliary inversion (spaCy)")
                        return True
            
            # Rule 5: Check for WH-word at sentence start
            if len(doc) > 0:
                first_token = doc[0]
                if first_token.tag_ in ["WDT", "WP", "WP$", "WRB"]:  # WH-determiners, pronouns, adverbs
                    logger.debug(f"Question detected: WH-word '{first_token.text}' (spaCy)")
                    return True
            
            # Rule 6: Check sentence structure (subject-auxiliary inversion)
            # E.g., "Are you coming?" vs "You are coming"
            has_aux_before_subj = False
            subj_idx = None
            aux_idx = None
            
            for token in doc:
                if token.dep_ in ["nsubj", "nsubjpass"] and subj_idx is None:
                    subj_idx = token.i
                if token.dep_ == "aux" and aux_idx is None:
                    aux_idx = token.i
            
            if aux_idx is not None and subj_idx is not None and aux_idx < subj_idx:
                has_aux_before_subj = True
                logger.debug(f"Question detected: subject-auxiliary inversion (spaCy)")
                return True
            
            # Rule 7: Check for modal verbs at start (Can/Could/Would/Should...)
            if len(doc) > 0 and doc[0].tag_ == "MD":  # Modal verb
                logger.debug(f"Question detected: modal verb '{doc[0].text}' at start (spaCy)")
                return True
        
        return False