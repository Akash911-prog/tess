import spacy
from spacy.tokens.doc import Doc
import sys

from configs.fillers import FILLERS
from configs.intents import INTENTS
from configs.patters import params_pattern

class LCN():
    """
    Linguistic Command Normalizer (LCN). This class encapsulates the logic for matching user's natural language input to intents and also extracting needed params.
    """

    def __init__(self) -> None:
        self.nlp = self._load_model("en_core_web_lg")
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
            if token.pos_ in {"VERB", "NOUN", "NUM", "ADJ", "PROPN"} or token.dep_ == "prt":
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

        if best_score < 0.65:   # similarity threshold
            best_cmd = "unknown"

        return (best_cmd, text_doc.text)
    
    def extract_params(self, text: str) -> dict[str, str]:
        """
        Extracts parameters from the user's input text.

        Parameters are extracted by matching against predefined patterns using spaCy's entity ruler.
        The extracted parameters are stored in a dictionary where the keys are the parameter names and the values are the corresponding values extracted from the input text.

        Args:
            text (str): The user's input text.

        Returns:
            dict[str, str]: A dictionary containing the extracted parameters.

        Example:
            >>> lcn.extract_params("can you move a bit to the right?")
            {'direction': 'right', 'distance': 'a bit'}
        """
        
        # Add entity ruler to the NLP pipeline
        entity_ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        # Define the patterns to match for parameter extraction
        entity_ruler.add_patterns(params_pattern)
        # Preprocess the text
        text = self.preprocess(text)
        # Parse the text using the NLP pipeline
        doc: Doc = self.nlp(text)

        # Initialize a dictionary to store the extracted parameters
        params: dict[str, str] = {}

        # Iterate over the extracted parameters and store them in the dictionary
        for ent in doc.ents:
            params[ent.label_] = ent.text

        # Remove the entity ruler from the NLP pipeline
        self.nlp.remove_pipe("entity_ruler")

        # Return the extracted parameters
        return params
    
    def _load_model(self, model_name : str):
        try:
            nlp = spacy.load(model_name)
            return nlp
        
        except OSError:
            print(f"Model not found: {model_name}")
            # Use spacy.cli.download to programmatically download the model
            try:
                spacy.cli.download(model_name)
                print(f"Successfully downloaded {model_name}.")
                
                # 3. Reload the model after successful download
                nlp = spacy.load(model_name)
                print("Model loaded successfully after download!")
                return nlp
                
            except Exception as e:
                # Handle potential errors during the download process itself
                print(f"Error during model download: {e}", file=sys.stderr)
                sys.exit(1)

    def normalize(self, text: str) -> dict[str, str | dict[str, str] | None]:
        """
        Normalizes the given text into a dictionary containing the intent and parameters.

        Args:
            text (str): The text to be normalized.

        Returns:
            dict[str, str | dict[str, str] | None]: A dictionary containing the normalized intent and parameters.
                If the intent is unknown, returns a dictionary with intent as None and an empty parameters dictionary.
        """
        intent, _ = self.get_intent(text)

        if intent == "code":
            return {
                "intent": intent,
                "params": {
                    "query": text
                }
            }

        if intent != 'unknown':
            params = self.extract_params(text)
            return {
                "intent": intent,
                "params": params
            }
        
        return {
            "intent" : None,
            "params" : {}
        }

if __name__ == "__main__":
    lcn = LCN()
    print(lcn.extract_params("open file"))
    

