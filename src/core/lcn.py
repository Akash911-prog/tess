from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from typing import Optional
import numpy as np
import logging
import pickle
from pathlib import Path


from configs.fillers import FILLERS
from configs.intents import INTENTS
from configs.patters import params_pattern

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    logging.warning("rapidfuzz not installed. Install with 'pip install rapidfuzz' for better accuracy.")


logger = logging.getLogger('tess.lcn')


@dataclass
class IntentResult:
    """Result from intent matching with detailed scoring information"""
    intent: str
    cleaned_text: str
    params: dict
    confidence: float
    
    # Score breakdown
    semantic_score: float
    lexical_score: float
    overlap_boost: float
    
    # Ambiguity detection
    second_best_intent: Optional[str] = None
    second_best_score: Optional[float] = None
    
    @property
    def is_ambiguous(self) -> bool:
        """Check if intent matching was ambiguous (top 2 within 0.1)"""
        if not self.second_best_score:
            return False
        return (self.confidence - self.second_best_score) < 0.1
    
    @property
    def match_quality(self) -> str:
        """Categorize match quality for easier handling"""
        if self.confidence >= 0.9:
            return "excellent"
        elif self.confidence >= 0.75:
            return "good"
        elif self.confidence >= 0.6:
            return "acceptable"
        else:
            return "poor"


class LCN:
    """
    Linguistic Command Normalizer.
    Converts natural language into structured commands:
      - intent (open, copy, volume up, etc.)
      - parameters (numbers, directions, targets)
      - confidence score with breakdown
    """

    # Configuration constants
    SEMANTIC_WEIGHT = 0.6
    LEXICAL_WEIGHT = 0.3
    OVERLAP_BOOST_MAX = 0.2
    CONFIDENCE_THRESHOLD = 0.62
    PARAM_BOOST_MULTIPLIER = 1.1
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    AMBIGUITY_THRESHOLD = 0.1

    def __init__(self) -> None:
        logger.info('LCN initializing...')
        
        self.nlp = self._load_model("en_core_web_lg")
        logger.debug('spaCy model loaded')

        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.debug('SentenceTransformer model loaded')

        # Preload cleaned intent docs for similarity matching
        self.intent_embeddings = self._precompute_intent_embeddings()
        logger.debug(f'Generated intent docs for {len(self.intent_embeddings)} intents')
        
        # Store raw intent phrases for fuzzy matching
        self.intent_phrases = self._generate_intent_phrases()
        logger.debug(f'Generated intent phrases for {len(self.intent_phrases)} intents')

        # Create a persistent entity ruler ONCE
        if "entity_ruler" not in self.nlp.pipe_names:
            self.entity_ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            self.entity_ruler.add_patterns(params_pattern)
            logger.debug('Entity ruler added to pipeline')

        cache_file = Path("cache/embeddings.pkl")

        # Load cache
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                self.embedding_cache = pickle.load(f)
        else:
            self.embedding_cache = {}

        logger.info(f'LCN initialized | rapidfuzz={HAS_RAPIDFUZZ}')

    # ---------------------------------------------------------
    # PREPROCESSING
    # ---------------------------------------------------------
    def preprocess(self, text: str) -> str:
        """
        Lowercase and remove filler words.
        
        Args:
            text: Raw user input
            
        Returns:
            Preprocessed text with fillers removed
        """
        text = text.lower().strip()

        # Remove fillers in the middle
        for filler in FILLERS:
            text = text.replace(f" {filler} ", " ")
        
        # Clean up edge cases (start/end)
        text = text.strip()
        for filler in FILLERS:
            if text.startswith(filler + " "):
                text = text[len(filler):].strip()
            if text.endswith(" " + filler):
                text = text[:-len(filler)].strip()

        return " ".join(text.split())

    # ---------------------------------------------------------
    # CLEAN COMMAND
    # ---------------------------------------------------------
    def clean_command(self, text: str) -> str:
        """
        Remove filler tokens and keep only action-relevant tokens.
        Uses spaCy's linguistic features for context preservation.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Cleaned text with only relevant tokens
        """
        doc = self.nlp(text)

        # POS tags to keep
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
    def _precompute_intent_embeddings(self):
        """Precompute embeddings for all intents in batch"""
        intent_map = {}
        
        for intent, phrases in INTENTS.items():
            cleaned_phrases = []
            for phrase in phrases:
                processed = self.preprocess(phrase)
                cleaned = self.clean_command(processed)
                if cleaned:
                    cleaned_phrases.append(cleaned)
            
            # Batch encode all phrases for this intent
            if cleaned_phrases:
                embeddings = self.embedding_model.encode(
                    cleaned_phrases,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                intent_map[intent] = (cleaned_phrases, embeddings)
        
        return intent_map
    
    def _generate_intent_phrases(self) -> dict[str, list[str]]:
        """
        Store cleaned string versions of intent phrases for fuzzy matching.
        
        Returns:
            Dict mapping intent names to lists of cleaned phrase strings
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
    def _vector(self, text: str):
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        embedding = self.embedding_model.encode(text)
        self.embedding_cache[text] = embedding
        return embedding

    def _sim(self, doc_vec, query_vec):
        """
        Fast cosine similarity with zero-vector handling.
        
        Args:
            doc_vec: Document vector
            query_vec: Query vector
            
        Returns:
            Cosine similarity score (0-1)
        """
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
        Detect if text is a question pattern using multiple signals.
        
        Args:
            text: Preprocessed text
            
        Returns:
            True if text appears to be a question/search query
        """
        text_lower = text.lower().strip()
        
        # Check for question mark
        if text_lower.endswith("?"):
            logger.debug("Question detected: ends with '?'")
            return True
        
        # Common question starters
        question_starters = [
            "what is", "what are", "what's", "whats",
            "who is", "who are", "who's", "whos",
            "when is", "when are", "when was", "when's",
            "where is", "where are", "where's",
            "how do", "how does", "how can", "how to",
            "why is", "why are", "why does",
            "tell me", "show me", "give me",
            "find out", "look up", "search for"
        ]
        
        for starter in question_starters:
            if text_lower.startswith(starter):
                logger.debug(f"Question detected: starts with '{starter}'")
                return True
        
        # Check first 2 words for question words
        words = text_lower.split()[:2]
        question_words = {"what", "who", "when", "where", "why", "how", "is", "are", "can", "could"}
        if any(word in question_words for word in words):
            logger.debug(f"Question detected: contains question word in first 2 words")
            return True
        
        return False

    # ---------------------------------------------------------
    # INTENT MATCHING WITH HYBRID APPROACH
    # ---------------------------------------------------------
    def get_intent(self, text: str) -> IntentResult:
        """
        Match user input to an intent using hybrid approach:
        1. Question pattern detection (rule-based)
        2. Semantic similarity (spaCy vectors)
        3. Lexical similarity (fuzzy matching)
        4. Keyword overlap boosting
        
        Args:
            text: Raw user input
            
        Returns:
            IntentResult with intent, confidence, and scoring details
        """
        # Handle empty input
        if not text or not text.strip():
            logger.warning("Empty text input received")
            return IntentResult(
                intent="unknown",
                cleaned_text="",
                params={},
                confidence=0.0,
                semantic_score=0.0,
                lexical_score=0.0,
                overlap_boost=1.0
            )

        logger.debug(f"Processing input: '{text}'")
        
        # Preprocess
        processed = self.preprocess(text)
        logger.debug(f"After preprocessing: '{processed}'")
        
        # Rule-based: Check for question patterns
        if self._is_question_pattern(processed):
            logger.info(f"Question pattern detected: '{text}' -> search")
            return IntentResult(
                intent="search",
                cleaned_text=processed,
                params={},
                confidence=0.95,
                semantic_score=0.0,
                lexical_score=0.0,
                overlap_boost=1.0
            )
        
        # Clean for semantic matching
        cleaned = self.clean_command(processed)
        logger.debug(f"After cleaning: '{cleaned}'")
        
        if not cleaned:
            logger.warning(f"Empty text after cleaning: '{text}'")
            return IntentResult(
                intent="unknown",
                cleaned_text="",
                params={},
                confidence=0.0,
                semantic_score=0.0,
                lexical_score=0.0,
                overlap_boost=1.0
            )
        
        # Semantic matching
        query_vec = self._vector(cleaned)
        query_words = set(cleaned.split())
        
        # Track all matches for finding second-best
        all_scores = []  # List of (intent, combined_score, sem, lex, boost)
        
        for intent, docs in self.intent_embeddings.items():
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
                    overlap_boost = 1.0 + (self.OVERLAP_BOOST_MAX * word_overlap)
                else:
                    overlap_boost = 1.0
                
                # Combine scores with weights
                if HAS_RAPIDFUZZ:
                    combined_score = (
                        self.SEMANTIC_WEIGHT * sem_score + 
                        self.LEXICAL_WEIGHT * lex_score
                    ) * overlap_boost
                else:
                    combined_score = sem_score * overlap_boost
                
                all_scores.append((intent, combined_score, sem_score, lex_score, overlap_boost))
        
        # Sort by score (descending)
        all_scores.sort(key=lambda x: x[1], reverse=True)
        
        if not all_scores:
            logger.error("No intent matches found (should not happen)")
            return IntentResult(
                intent="unknown",
                cleaned_text=cleaned,
                params={},
                confidence=0.0,
                semantic_score=0.0,
                lexical_score=0.0,
                overlap_boost=1.0
            )
        
        # Get best and second-best
        best_intent, best_score, best_sem, best_lex, best_boost = all_scores[0]
        second_best = all_scores[1] if len(all_scores) > 1 else None
        
        # Apply confidence threshold
        if best_score < self.CONFIDENCE_THRESHOLD:
            logger.warning(
                f"Below threshold | input='{text}' | "
                f"best={best_intent}({best_score:.3f}) | "
                f"threshold={self.CONFIDENCE_THRESHOLD} | "
                f"sem={best_sem:.3f} lex={best_lex:.3f} boost={best_boost:.2f}"
            )
            
            return IntentResult(
                intent="unknown",
                cleaned_text=cleaned,
                params={},
                confidence=0.0,
                semantic_score=best_sem,
                lexical_score=best_lex,
                overlap_boost=best_boost,
                second_best_intent=second_best[0] if second_best else None,
                second_best_score=second_best[1] if second_best else None
            )
        
        # Create result
        result = IntentResult(
            intent=best_intent,
            cleaned_text=cleaned,
            params={},  # Will be filled by normalize()
            confidence=best_score,
            semantic_score=best_sem,
            lexical_score=best_lex,
            overlap_boost=best_boost,
            second_best_intent=second_best[0] if second_best else None,
            second_best_score=second_best[1] if second_best else None
        )
        
        # Log based on match quality
        if result.is_ambiguous:
            logger.warning(
                f"Ambiguous match | input='{text}' | "
                f"best={best_intent}({best_score:.3f}) | "
                f"second={result.second_best_intent}({result.second_best_score:.3f}) | "
                f"diff={best_score - result.second_best_score:.3f}"
            )
        else:
            logger.info(
                f"Intent matched | input='{text}' | "
                f"intent={best_intent} | confidence={best_score:.3f} | "
                f"quality={result.match_quality}"
            )
        
        return result

    # ---------------------------------------------------------
    # PARAMETER EXTRACTION
    # ---------------------------------------------------------
    def extract_params(self, text: str) -> dict[str, str]:
        """
        Extract parameters using entity ruler and spaCy NER.
        Normalized to lowercase keys for consistency.
        
        Args:
            text: Raw user input
            
        Returns:
            Dict of parameter_name -> value (lowercase keys)
        """
        text = self.preprocess(text)
        doc = self.nlp(text)

        params = {}
        
        # Extract from entity ruler and NER (normalize to lowercase)
        for ent in doc.ents:
            key = ent.label_.lower()
            params[key] = ent.text
        
        # Fallback: extract potential targets from noun chunks
        if "target" not in params and "app" not in params:
            noun_chunks = [chunk.text for chunk in doc.noun_chunks]
            if noun_chunks:
                params["target"] = noun_chunks[-1]
        
        logger.debug(f"Extracted params: {params}")
        return params

    # ---------------------------------------------------------
    # CONFIDENCE CALCULATION
    # ---------------------------------------------------------
    def _calculate_confidence(self, score: float, intent: str, params: dict) -> float:
        """
        Calculate overall confidence with parameter completeness boost.
        
        Args:
            score: Base confidence score from intent matching
            intent: Matched intent name
            params: Extracted parameters
            
        Returns:
            Adjusted confidence score (capped at 1.0)
        """
        confidence = score
        
        # Boost if we found expected parameters (lowercase keys)
        if intent in ["open", "close", "click"]:
            if params.get("target") or params.get("app"):
                confidence = min(confidence * self.PARAM_BOOST_MULTIPLIER, 1.0)
                logger.debug(f"Confidence boosted for {intent} with params")
        
        elif intent in ["volume up", "volume down"]:
            if params.get("amount"):
                confidence = min(confidence * self.PARAM_BOOST_MULTIPLIER, 1.0)
                logger.debug(f"Confidence boosted for {intent} with amount")
        
        return round(confidence, 3)

    # ---------------------------------------------------------
    # MODEL LOADING
    # ---------------------------------------------------------
    def _load_model(self, model_name: str):
        """
        Load spaCy model and auto-download if missing.
        
        Args:
            model_name: Name of spaCy model (e.g., "en_core_web_lg")
            
        Returns:
            Loaded spaCy model
        """
        try:
            import spacy
            nlp = spacy.load(model_name)
            logger.debug(f"Loaded spaCy model: {model_name}")
            return nlp

        except OSError:
            logger.info(f"Model {model_name} not found, downloading...")
            import spacy.cli
            spacy.cli.download(model_name)
            return spacy.load(model_name)

    # ---------------------------------------------------------
    # NORMALIZATION → final output
    # ---------------------------------------------------------
    def normalize(self, text: str) -> IntentResult:
        """
        Main entry point: normalize user input to structured command.
        
        Args:
            text: Raw user input
            
        Returns:
            IntentResult with intent, parameters, and confidence
        """
        logger.debug(f"normalize() called with: '{text}'")
        
        # Get intent
        result = self.get_intent(text)
        
        # Return early for unknown or search
        if result.intent in ["unknown", "search"]:
            if result.intent == "search":
                result.params = {"query": text}
            return result
        
        # Extract parameters for other intents
        result.params = self.extract_params(text)
        
        # Calculate final confidence with parameter boost
        result.confidence = self._calculate_confidence(
            result.confidence,
            result.intent,
            result.params
        )
        
        logger.info(
            f"normalize() complete | intent={result.intent} | "
            f"confidence={result.confidence:.3f} | params={result.params}"
        )
        
        return result