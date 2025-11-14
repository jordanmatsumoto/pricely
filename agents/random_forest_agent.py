# imports

import os
import re
from typing import List
from sentence_transformers import SentenceTransformer
import joblib
from agents.agent import Agent


class RandomForestAgent(Agent):

    name = "Random Forest Agent"
    color = Agent.MAGENTA

    def __init__(self):
        """
        Initialize this object by loading in the saved model weights
        and setting up lazy initialization for the SentenceTransformer vector encoding model
        """
        self.log("Random Forest Agent is initializing")

        # Load the trained Random Forest model immediately (safe)
        self.model = joblib.load('random_forest_model.pkl')

        # Lazy-load the SentenceTransformer later (prevents meta tensor errors)
        self.vectorizer = None

        self.log("Random Forest Agent is ready")

    def _load_vectorizer(self):
        """
        Load the SentenceTransformer model lazily and safely (CPU mode)
        """
        if self.vectorizer is None:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            self.log("Loading SentenceTransformer vectorizer lazily (CPU mode for safety)")
            self.vectorizer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')

    def price(self, description: str) -> float:
        """
        Use a Random Forest model to estimate the price of the described item
        :param description: the product to be estimated
        :return: the price as a float
        """
        self.log("Random Forest Agent is starting a prediction")

        # Ensure the SentenceTransformer is loaded before encoding
        self._load_vectorizer()

        # Encode the input description and predict price
        vector = self.vectorizer.encode([description])
        result = max(0, self.model.predict(vector)[0])

        self.log(f"Random Forest Agent completed - predicting ${result:.2f}")
        return result