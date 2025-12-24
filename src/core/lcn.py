from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from typing import Optional, Tuple
import numpy as np
import logging
import pickle
from pathlib import Path
import atexit


from configs.fillers import FILLERS
from configs.intents import INTENTS
from configs.patters import params_pattern
from configs.lcn_configs import (
    SEMANTIC_WEIGHT,
    LEXICAL_WEIGHT,
    OVERLAP_BOOST_MAX,
    CONFIDENCE_THRESHOLD,
    PARAM_BOOST_MULTIPLIER,
    MODEL_NAME,
    AMBIGUITY_THRESHOLD,
    SPACY_MODEL_NAME,
    CACHE_DIR
)
from libs.meta_request_detector import MetaRequestDetector
from libs.question_detector import QuestionDetector

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    logging.warning("rapidfuzz not installed. Install with 'pip install rapidfuzz' for better accuracy.")


logger = logging.getLogger('tess.core.lcn')


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
        return (self.confidence - self.second_best_score) < AMBIGUITY_THRESHOLD
    
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
    Converts natural language into structured commands using:
      - sentence-transformers for semantic understanding
      - spaCy for linguistic analysis
      - rapidfuzz for lexical matching
    """

    def __init__(self, model: str = MODEL_NAME) -> None:
        logger.info('LCN initializing...')
        
        # spaCy for NLP tasks (POS tagging, NER, cleaning)
        self.nlp = self._load_spacy_model(SPACY_MODEL_NAME)
        logger.debug('spaCy model loaded')
        EMBEDDING_MODEL_NAME = model

        # sentence-transformers for semantic embeddings
        self.embedding_model = self._load_embedding_model(EMBEDDING_MODEL_NAME)
        logger.debug(f'SentenceTransformer model loaded: {EMBEDDING_MODEL_NAME}')

        # Setup embedding cache
        self._setup_cache()

        # Precompute intent embeddings (done once at startup)
        self.intent_embeddings = self._precompute_intent_embeddings()
        logger.debug(f'Precomputed embeddings for {len(self.intent_embeddings)} intents')
        
        # Store cleaned phrases for fuzzy matching
        self.intent_phrases = self._generate_intent_phrases()
        logger.debug(f'Generated intent phrases for {len(self.intent_phrases)} intents')

        # Create entity ruler for parameter extraction
        if "entity_ruler" not in self.nlp.pipe_names:
            self.entity_ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            self.entity_ruler.add_patterns(params_pattern)
            logger.debug('Entity ruler added to pipeline')

        # Register cleanup on exit
        atexit.register(self._save_cache)

        self.questionDetector = QuestionDetector(self.nlp)
        self.metaRequestDetector = MetaRequestDetector(self.nlp)

        logger.info(f'LCN initialized | rapidfuzz={HAS_RAPIDFUZZ}')

    # ---------------------------------------------------------
    # MODEL LOADING
    # ---------------------------------------------------------
    def _load_spacy_model(self, model_name: str):
        """Load spaCy model (used for NLP tasks, not embeddings)"""
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

    def _load_embedding_model(self, model_name: str):
        """Load sentence-transformers model for semantic embeddings"""
        try:
            model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    # ---------------------------------------------------------
    # CACHE MANAGEMENT
    # ---------------------------------------------------------
    def _setup_cache(self):
        """Setup embedding cache for query vectors"""
        # Create cache directory
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_file = CACHE_DIR / "query_embeddings.pkl"
        
        # Load existing cache
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded embedding cache: {len(self.embedding_cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, starting fresh")
                self.embedding_cache = {}
        else:
            self.embedding_cache = {}
            logger.debug("Starting with empty embedding cache")

    def _save_cache(self):
        """Save embedding cache to disk"""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Saved embedding cache: {len(self.embedding_cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    # ---------------------------------------------------------
    # PREPROCESSING
    # ---------------------------------------------------------
    def _preprocess(self, text: str) -> str:
        """Lowercase and remove filler words"""
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
    def _clean_command(self, text: str) -> str:
        """Remove filler tokens, keep action-relevant tokens using spaCy"""
        doc = self.nlp(text)

        keep_pos = {"VERB", "NOUN", "NUM", "ADV", "PROPN", "ADJ"}
        directional_preps = {"up", "down", "to", "in", "on", "off", "out"}
        
        cleaned = []

        for tok in doc:
            if tok.is_space or tok.is_punct:
                continue
            if tok.text.lower() in FILLERS:
                continue
            if tok.pos_ == "ADP" and tok.text.lower() in directional_preps:
                cleaned.append(tok.text.lower())
                continue
            if tok.dep_ == "prt":
                cleaned.append(tok.lemma_)
                continue
            if tok.pos_ in keep_pos:
                cleaned.append(tok.lemma_)

        # Remove consecutive duplicates
        result = []
        for word in cleaned:
            if not result or result[-1] != word:
                result.append(word)

        return " ".join(result)

    # ---------------------------------------------------------
    # INTENT EMBEDDING GENERATION
    # ---------------------------------------------------------
    def _precompute_intent_embeddings(self) -> dict[str, Tuple[list[str], np.ndarray]]:
        """
        Precompute embeddings for all intent phrases.
        Returns dict mapping intent -> (phrases, embeddings)
        """
        intent_map = {}
        
        for intent, phrases in INTENTS.items():
            # Clean all phrases
            cleaned_phrases = []
            for phrase in phrases:
                processed = self._preprocess(phrase)
                cleaned = self._clean_command(processed)
                if cleaned:
                    cleaned_phrases.append(cleaned)
            
            if not cleaned_phrases:
                logger.warning(f"No valid phrases for intent: {intent}")
                continue
            
            # Batch encode all phrases for this intent
            logger.debug(f"Encoding {len(cleaned_phrases)} phrases for intent: {intent}")
            embeddings = self.embedding_model.encode(
                cleaned_phrases,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32
            )
            
            intent_map[intent] = (cleaned_phrases, embeddings)
            logger.debug(f"Intent '{intent}': {len(cleaned_phrases)} phrases, embedding shape: {embeddings.shape}")
        
        return intent_map
    
    def _generate_intent_phrases(self) -> dict[str, list[str]]:
        """Extract just the phrases from intent_embeddings for fuzzy matching"""
        return {
            intent: phrases 
            for intent, (phrases, _) in self.intent_embeddings.items()
        }

    # ---------------------------------------------------------
    # VECTOR COMPUTATION
    # ---------------------------------------------------------
    def _get_query_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for query text with caching.
        
        Args:
            text: Cleaned query text
            
        Returns:
            Embedding vector (numpy array)
        """
        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Compute embedding
        embedding = self.embedding_model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Cache it
        self.embedding_cache[text] = embedding
        
        # Periodically save cache (every 100 new entries)
        if len(self.embedding_cache) % 100 == 0:
            self._save_cache()
        
        return embedding

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score (0-1)
        """
        # Handle zero vectors
        if not vec1.any() or not vec2.any():
            return 0.0
        
        # Compute norms
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if norm_product == 0:
            return 0.0
        
        # Cosine similarity
        return float(np.dot(vec1, vec2) / norm_product)
    
    # ---------------------------------------------------------
    # INTENT MATCHING
    # ---------------------------------------------------------
    def _get_intent(self, text: str) -> IntentResult:
        """
        Match user input to an intent using hybrid approach:
        1. Rule-based question detection
        2. Semantic similarity (sentence-transformers)
        3. Lexical similarity (fuzzy matching)
        4. Keyword overlap boosting
        """
        # Handle empty input
        if not text or not text.strip():
            logger.warning("Empty text input received")
            return IntentResult(
                intent="unknown", cleaned_text="", params={},
                confidence=0.0, semantic_score=0.0, lexical_score=0.0, overlap_boost=1.0
            )

        logger.debug(f"Processing input: '{text}'")
        
        # _Preprocess
        processed = self._preprocess(text)
        logger.debug(f"After _preprocessing: '{processed}'")
        
        # Rule-based: Question detection
        if self.questionDetector._is_question_pattern(processed):
            logger.info(f"Question pattern detected: '{text}' -> non_executable")
            return IntentResult(
                intent="non_executable", cleaned_text=processed, params={},
                confidence=0.95, semantic_score=0.0, lexical_score=0.0, overlap_boost=1.0
            )

        if self.metaRequestDetector.is_meta_request(processed):
            logger.info(f"Meta request detected: '{text}' -> non_executable")
            return IntentResult(
                intent="non_executable", cleaned_text=processed, params={},
                confidence=0.95, semantic_score=0.0, lexical_score=0.0, overlap_boost=1.0
            )
        
        # Clean for semantic matching
        cleaned = self._clean_command(processed)
        logger.debug(f"After cleaning: '{cleaned}'")
        
        if not cleaned:
            logger.warning(f"Empty text after cleaning: '{text}'")
            return IntentResult(
                intent="unknown", cleaned_text="", params={},
                confidence=0.0, semantic_score=0.0, lexical_score=0.0, overlap_boost=1.0
            )
        
        # Get query embedding
        query_embedding = self._get_query_embedding(cleaned) 
        query_words = set(cleaned.split())
        
        # Track all matches
        all_scores = []  # (intent, combined_score, sem, lex, boost)
        
        # Compare against all intent embeddings
        for intent, (phrases, embeddings) in self.intent_embeddings.items():
            for i, (phrase, phrase_embedding) in enumerate(zip(phrases, embeddings)):
                # 1. Semantic similarity
                sem_score = self._cosine_similarity(phrase_embedding, query_embedding)
                
                # 2. Lexical similarity
                lex_score = 0.0
                if HAS_RAPIDFUZZ:
                    lex_score = fuzz.ratio(cleaned, phrase) / 100.0
                
                # 3. Keyword overlap boost
                phrase_words = set(phrase.split())
                if query_words and phrase_words:
                    overlap = len(query_words & phrase_words) / max(len(query_words), 1)
                    overlap_boost = 1.0 + (OVERLAP_BOOST_MAX * overlap)
                else:
                    overlap_boost = 1.0
                
                # Combine scores
                if HAS_RAPIDFUZZ:
                    combined = (
                        SEMANTIC_WEIGHT * sem_score + 
                        LEXICAL_WEIGHT * lex_score
                    ) * overlap_boost
                else:
                    combined = sem_score * overlap_boost
                
                all_scores.append((intent, combined, sem_score, lex_score, overlap_boost))
        
        # Sort by score
        all_scores.sort(key=lambda x: x[1], reverse=True)
        
        if not all_scores:
            logger.error("No intent matches found")
            return IntentResult(
                intent="unknown", cleaned_text=cleaned, params={},
                confidence=0.0, semantic_score=0.0, lexical_score=0.0, overlap_boost=1.0
            )
        
        # Get best and second-best
        best_intent, best_score, best_sem, best_lex, best_boost = all_scores[0]
        second_best = all_scores[1] if len(all_scores) > 1 else None
        
        # Threshold check
        if best_score < CONFIDENCE_THRESHOLD:
            logger.warning(
                f"Below threshold | input='{text}' | best={best_intent}({best_score:.3f}) | "
                f"threshold={CONFIDENCE_THRESHOLD} | sem={best_sem:.3f} lex={best_lex:.3f}"
            )
            return IntentResult(
                intent="unknown", cleaned_text=cleaned, params={},
                confidence=0.0, semantic_score=best_sem, lexical_score=best_lex,
                overlap_boost=best_boost,
                second_best_intent=second_best[0] if second_best else None,
                second_best_score=second_best[1] if second_best else None
            )
        
        # Create result
        result = IntentResult(
            intent=best_intent, cleaned_text=cleaned, params={},
            confidence=best_score, semantic_score=best_sem, lexical_score=best_lex,
            overlap_boost=best_boost,
            second_best_intent=second_best[0] if second_best else None,
            second_best_score=second_best[1] if second_best else None
        )
        
        # Log
        if result.is_ambiguous:
            logger.warning(
                f"Ambiguous | input='{text}' | best={best_intent}({best_score:.3f}) | "
                f"second={result.second_best_intent}({result.second_best_score:.3f})"
            )
        else:
            logger.info(
                f"Matched | input='{text}' | intent={best_intent} | "
                f"confidence={best_score:.3f} | quality={result.match_quality}"
            )
        
        return result

    # ---------------------------------------------------------
    # PARAMETER EXTRACTION
    # ---------------------------------------------------------
    def _extract_params(self, text: str) -> dict[str, str]:
        """Extract parameters using spaCy NER and entity ruler"""
        text = self._preprocess(text)
        doc = self.nlp(text)

        params = {}
        
        # Extract from NER (lowercase keys)
        for ent in doc.ents:
            key = ent.label_.lower()
            params[key] = ent.text
        
        # Fallback: noun chunks
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
        """Boost confidence if expected parameters are found"""
        confidence = score
        
        if intent in ["open", "close", "click"]:
            if params.get("target") or params.get("app"):
                confidence = min(confidence * PARAM_BOOST_MULTIPLIER, 1.0)
                logger.debug(f"Confidence boosted for {intent}")
        
        elif intent in ["volume up", "volume down"]:
            if params.get("amount"):
                confidence = min(confidence * PARAM_BOOST_MULTIPLIER, 1.0)
                logger.debug(f"Confidence boosted for {intent}")
        
        return round(confidence, 3)

    # ---------------------------------------------------------
    # MAIN API
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
        result = self._get_intent(text)
        
        # Handle special cases
        if result.intent in ["unknown", "non_executable"]:
            if result.intent == "non_executable":
                result.params = {"query": text}
            return result
        
        # Extract parameters
        result.params = self._extract_params(text)
        
        # Calculate final confidence
        result.confidence = self._calculate_confidence(
            result.confidence, result.intent, result.params
        )
        
        logger.info(
            f"normalize() complete | intent={result.intent} | "
            f"confidence={result.confidence:.3f} | params={result.params}"
        )
        
        return result