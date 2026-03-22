import pandas as pd
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from urllib.parse import urlparse
import torch
BASE_DIR = Path(__file__).resolve().parent
facts = pd.read_csv(BASE_DIR.parent / "fact.csv")
facts["claim"] = facts["claim"].astype(str).str.strip().str.lower()
facts = facts.drop_duplicates(subset=["claim"]).reset_index(drop=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
fact_texts = facts["claim"].tolist()
fact_embeddings = model.encode(
    fact_texts,
    convert_to_tensor=True,
    device=DEVICE
)
THRESHOLD_TRUE = 0.75
THRESHOLD_PARTIAL = 0.60
THRESHOLD_MIN = 0.45
def normalize(text):
    return re.sub(r"\s+", " ", text.lower().strip())
def extract_claims(text):
    text = normalize(text)
    sentences = re.split(r"[.!?\n]", text)
    claims = []
    for s in sentences:
        s = s.strip()
        if len(s) > 15:
            claims.append(s)
    return claims if claims else [text]
def detail_mismatch_penalty(claim, fact):
    claim = claim.lower()
    fact = fact.lower()
    risky_keywords = [
        "terrorist", "assassinated", "bomb", "attack",
        "rape", "murdered", "shot dead", "conspiracy",
        "hoax", "fake"
    ]
    for word in risky_keywords:
        if word in claim and word not in fact:
            return True
    return False
def entity_mismatch(claim, fact):
    entities = ["delhi", "pakistan", "iran", "israel"]
    for e in entities:
        if e in claim and e not in fact:
            return True
    return False
def generate_reason(status, match_count):
    if status == "True":
        return f"{match_count} supporting facts found."
    elif status == "False":
        return "Contradictory facts detected."
    elif status == "Partially True / Misleading":
        return "Contains unsupported or high-risk details."
    else:
        return "No reliable match found."
def get_credibility_score(url):
    if not url:
        return 50
    domain = urlparse(url).netloc.lower()
    if any(x in domain for x in ["gov", "edu"]):
        return 90
    elif any(x in domain for x in ["bbc", "reuters"]):
        return 85
    elif any(x in domain for x in ["ndtv", "cnn", "news"]):
        return 75
    else:
        return 50
def overall_verdict(results):
    true_count = sum(1 for r in results if r["status"] == "True")
    false_count = sum(1 for r in results if r["status"] == "False")
    if false_count > 0:
        return "Likely Misinformation"
    elif true_count == len(results) and len(results) > 0:
        return "Highly Accurate"
    elif true_count > 0:
        return "Partially True / Misleading"
    else:
        return "Unverified"
def verify_article(text, url=None):
    if not text or len(text.strip()) < 10:
        return {
            "claims": [],
            "truth_percentage": 0,
            "credibility_score": get_credibility_score(url),
            "confidence_graph": [],
            "final_verdict": "Unverified"
        }
    claims = extract_claims(text)
    claim_embeddings = model.encode(
        claims,
        convert_to_tensor=True,
        device=DEVICE
    )
    scores = util.cos_sim(claim_embeddings, fact_embeddings)
    results = []
    confidence_graph = []
    score_sum = 0
    for i, claim in enumerate(claims):
        score_row = scores[i]
        top_k = torch.topk(score_row, k=3)
        top_scores = top_k.values.cpu().numpy()
        top_indices = top_k.indices.cpu().numpy()
        relevant_matches = [
            facts.iloc[idx]
            for j, idx in enumerate(top_indices)
            if top_scores[j] >= THRESHOLD_PARTIAL
        ]
        true_matches = sum(1 for f in relevant_matches if f["label"] == 1)
        false_matches = sum(1 for f in relevant_matches if f["label"] == 0)
        best_score = float(top_scores[0])
        confidence = round(float(top_scores.mean()) * 100, 2)
        best_fact = facts.iloc[top_indices[0]] if len(top_indices) > 0 else None
        best_fact_text = best_fact["claim"] if best_fact is not None else ""
        if best_score < THRESHOLD_MIN:
            status = "Unverified"
        elif detail_mismatch_penalty(claim, best_fact_text) or entity_mismatch(claim, best_fact_text):
            status = "Partially True / Misleading"
        elif true_matches > 0 and false_matches == 0:
            status = "True"
        elif false_matches > 0 and true_matches == 0:
            status = "False"
        elif true_matches > 0 and false_matches > 0:
            status = "Partially True / Misleading"
        else:
            status = "Unverified"
        if status == "True":
            score_sum += 1
        elif status == "Partially True / Misleading":
            score_sum += 0.5
        else:
            score_sum += 0
        results.append({
            "claim": claim,
            "status": status,
            "reason": generate_reason(status, len(relevant_matches)),
            "matched_fact": best_fact["claim"] if best_fact is not None else None,
            "correct_fact": best_fact["correct_fact"] if best_fact is not None else None,
            "source": best_fact["source"] if best_fact is not None else None,
            "confidence": confidence
        })
        confidence_graph.append({
            "claim": claim[:50],
            "confidence": confidence
        })
    truth_percentage = (score_sum / len(results)) * 100 if results else 0
    return {
        "claims": results,
        "truth_percentage": round(truth_percentage, 2),
        "credibility_score": get_credibility_score(url),
        "confidence_graph": confidence_graph,
        "final_verdict": overall_verdict(results)
    }