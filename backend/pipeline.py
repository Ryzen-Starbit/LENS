import pandas as pd
import re
from scaledown import compressor
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")
facts = pd.read_csv("fact.csv")
fact_embeddings = model.encode(facts["claim"].tolist(), convert_to_tensor=True)
def extract_claims(text):
    sentences = re.split(r"[.!?]", text)
    claims = []
    for s in sentences:
        s = s.strip()
        if len(s) > 25:
            claims.append(s)
    return claims
def verify_claim(claim):
    claim_embedding = model.encode(claim, convert_to_tensor=True)
    scores = util.cos_sim(claim_embedding, fact_embeddings)
    best_index = scores.argmax().item()
    fact = facts.iloc[best_index]
    return {
        "claim": claim,
        "is_true": bool(fact["label"]),
        "correct_fact": fact["correct_fact"],
        "source": fact["source"],
        "confidence": float(scores[0][best_index])
    }
def verify_article(text):
    compressed = compressor(text)
    claims = extract_claims(compressed)
    results = []
    true_count = 0
    for claim in claims:
        result = verify_claim(claim)
        results.append(result)
        if result["is_true"]:
            true_count += 1
    score = 0
    if len(claims) > 0:
        score = (true_count / len(claims)) * 100
    return {
        "truth_score": score,
        "claims": results
    }