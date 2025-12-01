import spacy
from spacy.tokens import Doc
from functools import lru_cache
import numpy as np

from configs.fillers import FILLERS
from configs.intents import INTENTS
from configs.patters import params_pattern

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    print("Warning: rapidfuzz not installed. Install with 'pip install rapidfuzz' for better accuracy.")


class LCN:
    """
    Improved Linguistic Command Normalizer.
    Converts natural language into:
      - intent (open, copy, volume up, etc.)
      - parameters (numbers, directions, targets…)
      - confidence score
    """

    def __init__(self) -> None:
        self.nlp = self._load_model("en_core_web_lg")

        # Preload cleaned intent docs for similarity matching
        self.intent_docs = self.generate_intent_docs()
        
        # Store raw intent phrases for fuzzy matching
        self.intent_phrases = self._generate_intent_phrases()

        # Create a persistent entity ruler ONCE
        if "entity_ruler" not in self.nlp.pipe_names:
            self.entity_ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            self.entity_ruler.add_patterns(params_pattern)

    # ---------------------------------------------------------
    # PREPROCESSING
    # ---------------------------------------------------------
    def preprocess(self, text: str) -> str:
        """Lowercase & remove filler words."""
        text = text.lower().strip()

        for f in FILLERS:
            # Use word boundaries to avoid partial matches
            text = text.replace(f" {f} ", " ")
        
        # Clean up edge cases
        text = text.strip()
        for f in FILLERS:
            if text.startswith(f + " "):
                text = text[len(f):].strip()
            if text.endswith(" " + f):
                text = text[:-len(f)].strip()

        return " ".join(text.split())

    # ---------------------------------------------------------
    # CLEAN COMMAND
    # ---------------------------------------------------------
    def clean_command(self, text: str) -> str:
        """
        Removes filler tokens and keeps action-relevant tokens.
        More aggressive cleaning with better context preservation.
        """
        doc = self.nlp(text)

        # Expanded POS tags for better context
        keep_pos = {"VERB", "NOUN", "NUM", "ADV", "PROPN", "ADJ"}
        
        # Important prepositions for directional commands
        directional_preps = {"up", "down", "to", "in", "on", "off", "out"}
        
        cleaned = []

        for tok in doc:
            if tok.is_space or tok.is_punct:
                continue

            if tok.text.lower() in FILLERS:
                continue

            # Keep important prepositions
            if tok.pos_ == "ADP" and tok.text.lower() in directional_preps:
                cleaned.append(tok.text.lower())
                continue
            
            # Keep particle verbs (e.g., "shut down")
            if tok.dep_ == "prt":
                cleaned.append(tok.lemma_)
                continue
            
            if tok.pos_ in keep_pos:
                cleaned.append(tok.lemma_)

        # Remove only consecutive duplicates (preserve order)
        result = []
        for word in cleaned:
            if not result or result[-1] != word:
                result.append(word)

        return " ".join(result)

    # ---------------------------------------------------------
    # INTENT DOC GENERATION
    # ---------------------------------------------------------
    def generate_intent_docs(self) -> dict[str, list[Doc]]:
        """
        Preload cleaned spaCy docs for all intent example phrases.
        This ensures consistency between query and intent processing.
        """
        intent_map = {}

        for intent, phrases in INTENTS.items():
            cleaned_phrases = []
            for phrase in phrases:
                processed = self.preprocess(phrase)
                cleaned = self.clean_command(processed)
                if cleaned:  # Only add non-empty cleaned phrases
                    cleaned_phrases.append(self.nlp(cleaned))
            
            intent_map[intent] = cleaned_phrases

        return intent_map
    
    def _generate_intent_phrases(self) -> dict[str, list[str]]:
        """
        Store cleaned string versions of intent phrases for fuzzy matching.
        """
        intent_phrases = {}
        
        for intent, phrases in INTENTS.items():
            cleaned_phrases = []
            for phrase in phrases:
                processed = self.preprocess(phrase)
                cleaned = self.clean_command(processed)
                if cleaned:
                    cleaned_phrases.append(cleaned)
            
            intent_phrases[intent] = cleaned_phrases
        
        return intent_phrases

    # ---------------------------------------------------------
    # CACHED SIMILARITY COMPUTATION
    # ---------------------------------------------------------
    @lru_cache(maxsize=5000)
    def _vector(self, text: str):
        """Cache expensive vector lookups."""
        return self.nlp(text).vector

    def _sim(self, doc_vec, query_vec):
        """Fast cosine similarity with zero-vector handling."""
        if not doc_vec.any() or not query_vec.any():
            return 0.0
        
        norm_product = np.linalg.norm(doc_vec) * np.linalg.norm(query_vec)
        if norm_product == 0:
            return 0.0
            
        return float(np.dot(doc_vec, query_vec) / norm_product)

    # ---------------------------------------------------------
    # QUESTION PATTERN DETECTION
    # ---------------------------------------------------------
    def _is_question_pattern(self, text: str) -> bool:
        """
        Detect if text is a question pattern (what/who/when/where/how/why).
        """
        question_starters = [
            "what is", "what are", "what's", "whats",
            "who is", "who are", "who's", "whos",
            "when is", "when are", "when was", "when's",
            "where is", "where are", "where's",
            "how do", "how does", "how can", "how to",
            "why is", "why are", "why does",
            "tell me", "show me", "give me",
            "find out", "look up"
        ]
        
        text_lower = text.lower().strip()
        for starter in question_starters:
            if text_lower.startswith(starter):
                return True
        
        return False

    # ---------------------------------------------------------
    # INTENT MATCHING WITH HYBRID APPROACH
    # ---------------------------------------------------------
    def get_intent(self, text: str) -> tuple[str, str, float]:
        """
        Computes the best matching intent using:
        1. Question pattern detection (for search)
        2. Semantic similarity (spaCy vectors)
        3. Lexical similarity (fuzzy matching)
        4. Keyword overlap boosting
        
        Returns: (intent, cleaned_text, confidence_score)
        """
        processed = self.preprocess(text)
        
        # Check for question patterns BEFORE cleaning
        if self._is_question_pattern(processed):
            return "search", processed, 0.95
        
        cleaned = self.clean_command(processed)
        
        if not cleaned:
            return "unknown", "", 0.0

        query_vec = self._vector(cleaned)
        query_words = set(cleaned.split())

        best_intent = "unknown"
        best_score = 0.0
        best_match_info = {}

        for intent, docs in self.intent_docs.items():
            intent_phrases = self.intent_phrases.get(intent, [])
            
            for i, doc in enumerate(docs):
                # 1. Semantic similarity
                sem_score = self._sim(doc.vector, query_vec)
                
                # 2. Lexical similarity (if rapidfuzz available)
                lex_score = 0.0
                if HAS_RAPIDFUZZ and i < len(intent_phrases):
                    lex_score = fuzz.ratio(cleaned, intent_phrases[i]) / 100.0
                
                # 3. Keyword overlap boost
                intent_words = set(doc.text.split())
                if query_words and intent_words:
                    word_overlap = len(query_words & intent_words) / max(len(query_words), 1)
                    overlap_boost = 1.0 + (0.2 * word_overlap)  # Up to 20% boost
                else:
                    overlap_boost = 1.0
                
                # Combine scores with weights
                if HAS_RAPIDFUZZ:
                    combined_score = (0.6 * sem_score + 0.3 * lex_score) * overlap_boost
                else:
                    combined_score = sem_score * overlap_boost
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_intent = intent
                    best_match_info = {
                        "semantic": sem_score,
                        "lexical": lex_score,
                        "overlap_boost": overlap_boost
                    }

        # Adaptive threshold based on score composition
        threshold = 0.62 if HAS_RAPIDFUZZ else 0.68
        
        if best_score < threshold:
            best_intent = "unknown"
            best_score = 0.0

        return best_intent, cleaned, best_score

    # ---------------------------------------------------------
    # PARAMETER EXTRACTION
    # ---------------------------------------------------------
    def extract_params(self, text: str) -> dict[str, str]:
        """
        Extract parameters using entity ruler and spaCy NER.
        Improved with noun chunk fallback for targets.
        """
        text = self.preprocess(text)
        doc = self.nlp(text)

        params = {}
        
        # Prioritize entity ruler and NER
        for ent in doc.ents:
            params[ent.label_] = ent.text
        
        # Fallback: extract potential targets from noun chunks
        if "target" not in params and "app" not in params:
            noun_chunks = [chunk.text for chunk in doc.noun_chunks]
            if noun_chunks:
                # Use the last noun chunk as likely target
                params["target"] = noun_chunks[-1]
        
        return params

    # ---------------------------------------------------------
    # CONFIDENCE CALCULATION
    # ---------------------------------------------------------
    def _calculate_confidence(self, score: float, intent: str, params: dict) -> float:
        """
        Calculate overall confidence based on:
        - Intent matching score
        - Parameter extraction completeness
        """
        confidence = score
        
        # Boost confidence if we found expected parameters
        if intent in ["open", "close", "click"]:
            if params.get("TARGET") or params.get("APP"):
                confidence = min(confidence * 1.1, 1.0)
        
        elif intent in ["volume up", "volume down"]:
            if params.get("AMOUNT"):
                confidence = min(confidence * 1.1, 1.0)
        
        return round(confidence, 3)

    # ---------------------------------------------------------
    # MODEL LOADING
    # ---------------------------------------------------------
    def _load_model(self, model_name: str):
        """
        Load spaCy model and auto-download if missing.
        """
        try:
            import spacy
            nlp = spacy.load(model_name)
            return nlp

        except OSError:
            print(f"Downloading {model_name}...")
            import spacy.cli
            spacy.cli.download(model_name)
            return spacy.load(model_name)

    # ---------------------------------------------------------
    # NORMALIZATION → final output
    # ---------------------------------------------------------
    def normalize(self, text: str) -> dict[str, str | dict[str, str] | float | None]:
        """
        Returns:
            {
                "intent": <intent or None>,
                "params": {...},
                "confidence": <float 0-1>,
                "cleaned": <cleaned query string>
            }
        """
        intent, cleaned, score = self.get_intent(text)

        # Special handling for code generation
        if intent == "code":
            return {
                "intent": "code",
                "params": {"query": text},
                "confidence": score,
                "cleaned": cleaned
            }

        if intent != "unknown":
            params = self.extract_params(text)
            confidence = self._calculate_confidence(score, intent, params)
            
            return {
                "intent": intent,
                "params": params,
                "confidence": confidence,
                "cleaned": cleaned
            }

        return {
            "intent": None,
            "params": {},
            "confidence": 0.0,
            "cleaned": cleaned
        }