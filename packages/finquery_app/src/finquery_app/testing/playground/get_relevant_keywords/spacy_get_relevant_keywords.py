import re
import time
from typing import List, Set

import spacy
import tiktoken
from spacy.language import Language
from tiktoken.core import Encoding


def count_tokens(text: str, encoder: Encoding) -> int:
    """Counts the number of tokens in a string using the provided tiktoken encoder."""
    return len(encoder.encode(text, disallowed_special=()))


def _filter_and_clean_keywords(candidates: Set[str], max_keywords: int, encoder: Encoding) -> List[str]:
    """Internal helper to filter, clean, and deduplicate keyword candidates."""
    default_stop_words = {"their", "this", "the", "our", "we", "if", "it", "its", "in", "a", "an", "for", "and", "or"}
    final_keywords, seen_lower = [], set()

    sorted_candidates = sorted(list(candidates), key=len, reverse=True)

    for keyword in sorted_candidates:
        kw_clean = keyword.strip(" “”)’'.,:()").replace('’', "'").strip()
        kw_lower = kw_clean.lower()

        if (not kw_clean or len(kw_clean) < 4 or kw_lower in default_stop_words or any(
            char.isdigit() for char in kw_clean) or count_tokens(kw_clean, encoder) > 7):
            continue

        is_redundant_substring = any(kw_lower in seen for seen in seen_lower)
        if not is_redundant_substring:
            final_keywords.append(kw_clean)
            seen_lower.add(kw_lower)

        if len(final_keywords) >= max_keywords:
            break
    return final_keywords


def batch_extract_nlp_keywords(list_of_texts: List[str], nlp_model: Language, encoder_model: Encoding,
        max_keywords_per_item: int = 15) -> List[List[str]]:
    """
    Extracts keywords from a BATCH of texts using preloaded spaCy and tiktoken models.
    """
    all_results = []
    cleaned_texts = [re.sub(r'^\s*([a-zA-Z0-9]+\.|-|\*)\s+', '', text, flags=re.MULTILINE) for text in list_of_texts]
    cleaned_texts = [' '.join(text.split()) for text in cleaned_texts]

    docs = nlp_model.pipe(cleaned_texts, batch_size=500)

    for doc in docs:
        candidates = set()
        for chunk in doc.noun_chunks:
            candidates.add(chunk.text)
        for ent in doc.ents:
            candidates.add(ent.text)

        keywords = _filter_and_clean_keywords(candidates, max_keywords_per_item, encoder=encoder_model)
        all_results.append(keywords)

    return all_results


if __name__ == '__main__':

    print("Loading NLP models into memory...")
    start_load_time = time.time()
    try:
        NLP_MODEL = spacy.load("en_core_web_sm")
        ENCODER_MODEL = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"Error loading models: {e}")
        exit()
    end_load_time = time.time()
    print(f"Models loaded in {end_load_time - start_load_time:.2f} seconds.\n")

    SAMPLE_TEXTS = [
        "The market for our platforms is rapidly evolving. Our future success will depend in large part on the growth and expansion of this market, which is difficult to predict and relies on a number of factors, including customer adoption, customer demand, changing customer needs, the entry of competitive products, the success of existing competitive products, and potential customers' willingness to adopt an alternative approach to data collection, storage, and processing.",
        "Changing the Company's operations in accordance with new or changed restrictions on international trade can be expensive, time-consuming and disruptive to the Company's operations. Such restrictions can be announced with little or no advance notice and the Company may not be able to effectively mitigate all adverse impacts from such measures. For example, tensions between governments, including the U.S. and China, have in the past led to tariffs and other restrictions being imposed on the Company's business.",
        "Substantially all of the Company's manufacturing is performed in whole or in part by outsourcing partners located primarily in China mainland, India, Japan, South Korea, Taiwan and Vietnam. This concentration of manufacturing is currently performed by a small number of outsourcing partners, often in single locations. While these arrangements can lower operating costs, they also reduce the Company's direct control over production and distribution.",
        "Masimo Corporation and Cercacor Laboratories, Inc. (together, 'Masimo') filed a complaint before the U.S. International Trade Commission (the 'ITC') alleging infringement by the Company of five patents relating to the functionality of the blood oxygen feature in Apple Watch Series 6 and 7. In its complaint, Masimo sought a permanent exclusion order prohibiting importation to the United States of certain Apple Watch models.",
        "The Company's new products often utilize custom components available from only one source. When a component or product uses new technologies, initial capacity constraints may exist until the suppliers' yields have matured or their manufacturing capacities have increased. The continued availability of these components at acceptable prices, or at all, can be affected for any number of reasons.",
        "The Company and its global supply chain are dependent on complex information technology systems and are exposed to information technology system failures or network disruptions caused by natural disasters, accidents, power disruptions, telecommunications failures, acts of terrorism or war, computer viruses, physical or electronic break-ins, ransomware or other cybersecurity incidents.",
        "As of September 30, 2023, the Company was authorized by the Board of Directors to purchase up to $90 billion of the Company's common stock under a share repurchase program announced on May 4, 2023. Under the programs, shares may be repurchased in privately negotiated or open market transactions, including under plans complying with Rule 10b5-1 under the Exchange Act.",
        "The Company issues unsecured short-term promissory notes pursuant to a commercial paper program. The Company uses net proceeds from the commercial paper program for general corporate purposes, including dividends and share repurchases. As of September 30, 2023 and September 24, 2022, the Company had $6.0 billion and $10.0 billion of commercial paper outstanding, respectively, with maturities generally less than nine months.",
        "The Company believes that compensation should be competitive and equitable, and should enable employees to share in the Company's success. The Company recognizes its people are most likely to thrive when they have the resources to meet their needs and the time and support to succeed in their professional and personal lives. In support of this, the Company offers a wide variety of benefits for employees around the world.",
        "Although the Company believes the ownership of such intellectual property rights is an important factor in differentiating its business and that its success does depend in part on such ownership, the Company relies primarily on the innovative skills, technical competence and marketing abilities of its personnel. The Company regularly files patent, design, copyright and trademark applications to protect innovations arising from its research, development, design and marketing."]

    NUM_CHUNKS_TO_TEST = 20000
    test_batch = SAMPLE_TEXTS * (NUM_CHUNKS_TO_TEST // len(SAMPLE_TEXTS))
    print(f"Prepared a test batch of {NUM_CHUNKS_TO_TEST} text chunks for measurement.\n")

    print("!!! --- !!! Running Performance Test !!! --- !!!")
    start_batch_time = time.time()

    batch_results = batch_extract_nlp_keywords(list_of_texts=test_batch, nlp_model=NLP_MODEL,
        encoder_model=ENCODER_MODEL)

    end_batch_time = time.time()

    total_batch_time_s = end_batch_time - start_batch_time
    total_batch_time_ms = total_batch_time_s * 1000
    avg_time_per_chunk_ms = total_batch_time_ms / len(test_batch)

    print("\n--- PERFORMANCE RESULTS ---")
    print(f"Total chunks processed: {len(test_batch)}")
    print(f"Total time taken: {total_batch_time_s:.2f} seconds")
    print(f"Average time per chunk: {avg_time_per_chunk_ms:.5f} ms")
    print("---------------------------\n")

    print("\nFirst sample text keywords:")
    print(batch_results[0])
    print("\nSecond sample text keywords:")
    print(batch_results[1])
