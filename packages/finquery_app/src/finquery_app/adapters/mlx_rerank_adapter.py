from typing import List
import mlx.core as mx
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
from finquery_app.config import MLX_RERANKING_MODEL_NAME

# Prototype written with the help of Perplexity. Here as a temporary solution until I can dig further into the
# documentation.
class MLXRerankAdapter(BaseCrossEncoder):

    def __init__(self, model_name: str = MLX_RERANKING_MODEL_NAME):
        print("Initializing MLX Reranker Adapter...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            print(f"✅ Successfully loaded MLX reranker model: {model_name}")
        except Exception as e:
            print(f"🔴 Failed to load MLX reranker model: {e}")
            raise

    def score(self, text_pairs: List[List[str]]) -> List[float]:
        """
        Scores a list of text pairs [query, document] and returns the relevance scores.

        This method is called by the LangChain compressor.

        Args:
            text_pairs: A list of pairs, where each pair is [query, document].

        Returns:
            A list of float scores corresponding to each text pair.
        """
        inputs = self.tokenizer(
            text_pairs,
            padding=True,
            truncation=True,
            return_tensors="np",
            max_length=512,
        )

        mlx_inputs = {key: mx.array(value) for key, value in inputs.items()}

        logits = self.model(**mlx_inputs).logits

        scores = logits.squeeze(-1)

        scores_list = scores.tolist()

        if isinstance(scores_list, float):
            return [scores_list]

        return scores_list

