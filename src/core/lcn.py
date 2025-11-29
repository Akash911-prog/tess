import spacy
from spacy.tokens.doc import Doc

from configs.fillers import FILLERS
from configs.intents import INTENTS

class LCN():
    """
    Linguistic Command Normalizer (LCN). This class encapsulates the logic for matching user's natural language input to intents and also extracting needed params.
    """

    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_lg")
        self.intent_docs = self.generate_intent_docs()

    def preprocess(self, text: str) -> str:
        """
        Preprocesses the user's input by removing filler words and converting to lowercase.
        
        Args:
            text (str): The user's input text.
        
        Returns:
            str: The preprocessed text.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove filler words
        for f in FILLERS:
            text = text.replace(f, "")
        
        # Strip leading and trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_command(self, text: str) -> str:
        """
        Cleans and normalizes a base command using spaCy linguistic features.

        Removes filler words, keeps verbs, nouns, numbers, and relevant modifiers.
        Converts to lemma form for normalization.

        Args:
            text (str): The user's input text to be cleaned and normalized.

        Returns:
            str: The cleaned and normalized command text.
        """
        doc = self.nlp(text.lower())

        cleaned_tokens = []

        # Iterate over each token in the input text
        for token in doc:
            # Skip filler words, punctuation, and spaces
            if token.text in FILLERS or token.is_punct or token.is_space:
                continue

            # Keep main verbs, nouns, numbers, and important adjectives
            # Verbs are the main actions in the command (e.g. "open", "close")
            # Nouns are the objects being acted upon (e.g. "item", "file")
            # Numbers are numerical values (e.g. "1", "2")
            # Adjectives are modifiers that describe the command (e.g. "up", "down")
            # "prt" is a dependency label for particles, which are small words that
            # are used to form phrases (e.g. "up", "down")
            if token.pos_ in {"VERB", "NOUN", "NUMBER", "ADJ", "PROPN"} or token.dep_ == "prt":
                cleaned_tokens.append(token.lemma_)

            # Special case to handle "up" and "down" as separate tokens
            if token.text.lower() in ['up', 'down'] and token.text.lower() not in cleaned_tokens:
                cleaned_tokens.append(token.text.lower())

        # Join the tokens back into a normalized string
        return " ".join(cleaned_tokens)
    
    def generate_intent_docs(self) -> dict[str, list[Doc]]:
        """
        Generates a mapping of commands to their corresponding intent documents.

        Args:
            None

        Returns:
            dict[str, list[Doc]]: A dictionary where the keys are command names and the values are lists of spaCy Doc objects representing the command's intent documents.
        """
        command_vector : dict = {}

        for command, phrases in INTENTS.items():
            command_vector[command] = [self.nlp(t) for t in phrases]

        return command_vector
    
    def get_intent(self, text: str) -> tuple[str, str]:
        """Utility that replicates your matching logic."""
        processed : str = self.preprocess(text)
        cleaned : str = self.clean_command(processed)
        text_doc : Doc = self.nlp(cleaned)

        best_cmd : str = ''
        best_score : float = 0.0

        for cmd, docs in self.intent_docs.items():
            for doc in docs:
                score = doc.similarity(text_doc)
                if score > best_score:
                    best_score = score
                    best_cmd = cmd

        if best_score < 0.55:   # similarity threshold
            best_cmd = "unknown"

        return (best_cmd, text_doc.text)