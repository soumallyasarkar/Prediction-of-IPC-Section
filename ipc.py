import os
import json
import pickle
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# -------------------------------
# CONFIG
# -------------------------------
IPC_FILE = "merged.json"
EMBED_FILE = "ipc_embeddings.pkl"
TFIDF_FILE = "ipc_tfidf.pkl"
# Using a larger model for better accuracy
MODEL_NAME = "all-mpnet-base-v2"  # More powerful than all-MiniLM-L6-v2
SIMILARITY_THRESHOLD = 0.35  # Minimum similarity score to consider
HYBRID_WEIGHT_EMBEDDING = 0.6  # Weight for semantic similarity
HYBRID_WEIGHT_KEYWORD = 0.4    # Weight for keyword similarity

# Load spaCy for NLP processing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model not found. Please install with: python -m spacy download en_core_web_sm")
    nlp = None

# -------------------------------
# SEVERITY CALIBRATION
# -------------------------------
def estimate_crime_severity(text):
    """Estimate the severity of a crime description based on keywords and context"""
    if not text:
        return 0.5  # Medium severity by default

    text_lower = text.lower()

    # Keywords indicating severity levels
    extreme_keywords = [
    'murder', 'kill', 'killed', 'stabbed to death', 'beheaded', 'hanged',
    'rape', 'raped', 'sexual assault', 'molested badly', 'gang rape',
    'terrorism', 'terrorist', 'bomb', 'explosive', 'blast', 'grenade',
    'shooting', 'firing', 'massacre', 'genocide',
    'kidnap', 'abduct', 'taken away', 'hostage', 'missing forcefully',
    'suicide', 'attempt suicide', 'jumped from', 'drank poison',
    'burn alive', 'acid attack'
]

    high_keywords = [
    'assault', 'attacked', 'beaten up', 'hit badly', 'punched',
    'hurt', 'injured', 'stab', 'knife attack', 'shot with gun',
    'rob', 'robbed', 'snatched bag', 'stole purse', 'chain snatching',
    'burglary', 'broke into house', 'looted', 'dacoity',
    'extort', 'demand money', 'blackmail', 'ransom',
    'threaten', 'life threat', 'warned to kill',
    'weapon', 'armed', 'gunpoint', 'knife point',
    'molested', 'touched inappropriately', 'forced himself',
    'arson', 'set fire', 'burnt house'
    ]

    medium_keywords = [
    'cheat', 'cheated', 'fraud', 'scam', 'took money', 'promised but ran away',
    'forgery', 'fake papers', 'counterfeit notes',
    'thief', 'pickpocket', 'wallet stolen', 'phone stolen',
    'steal', 'stole bike', 'cycle stolen', 'house theft',
    'embezzlement', 'took office money', 'bribery',
    'damage', 'broke window', 'damaged car', 'smashed property',
    'harass', 'harassed', 'eve teasing', 'catcalling',
    'defamation', 'spread rumors', 'insult publicly',
    'stalking', 'followed me', 'kept coming behind'
    ]

    low_keywords = [
    'trespass', 'entered property', 'jumped wall', 'came inside without permission',
    'nuisance', 'loud music', 'shouting on street',
    'disobey', 'not following rules', 'argued with officer',
    'disturb', 'disturbance', 'creating scene',
    'loitering', 'roaming suspiciously', 'wandering around',
    'simple hurt', 'slapped', 'pushed', 'minor injury',
    'public drinking', 'drunk on road', 'drunk fight',
    'eve teasing (mild)', 'passing comments', 'whistling at girls',
    'noise complaint', 'neighbors fighting loudly',
    'misconduct', 'misbehaved', 'abused in public'
]

    # Count occurrences
    extreme_count = sum(1 for word in extreme_keywords if word in text_lower)
    high_count = sum(1 for word in high_keywords if word in text_lower)
    medium_count = sum(1 for word in medium_keywords if word in text_lower)
    low_count = sum(1 for word in low_keywords if word in text_lower)

    # Calculate weighted severity score
    total = extreme_count + high_count + medium_count + low_count
    if total == 0:
        return 0.5  # Default medium severity

    severity_score = (
        extreme_count * 1.0 +
        high_count * 0.75 +
        medium_count * 0.5 +
        low_count * 0.25
    ) / total

    return min(severity_score, 1.0)  # Cap at 1.0

def estimate_punishment_severity(punishment_text):
    """Estimate the severity of punishment based on keywords"""
    if not punishment_text:
        return 0.5

    text_lower = punishment_text.lower()

    # Keywords indicating punishment severity
    extreme_keywords = ['death', 'life imprisonment', 'imprisonment for life']
    high_keywords = ['10 year', 'ten year', '14 year', 'fourteen year']
    medium_keywords = ['7 year', 'seven year', '5 year', 'five year', '3 year', 'three year']
    low_keywords = ['1 year', 'one year', '6 month', 'fine only', 'month']

    # Check for severity levels
    if any(word in text_lower for word in extreme_keywords):
        return 1.0
    elif any(word in text_lower for word in high_keywords):
        return 0.75
    elif any(word in text_lower for word in medium_keywords):
        return 0.5
    elif any(word in text_lower for word in low_keywords):
        return 0.25
    else:
        return 0.5  # Default medium severity

# -------------------------------
# TEXT PREPROCESSING
# -------------------------------
def preprocess_text(text):
    """Clean and preprocess text for better embeddings"""
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits but keep relevant legal terms
    text = re.sub(r'[^a-zA-Z\s§\-]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# -------------------------------
# LOAD MODEL
# -------------------------------
print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

# -------------------------------
# LOAD IPC DATA
# -------------------------------
with open(IPC_FILE, "r", encoding="utf-8") as f:
    ipc_data = json.load(f)

ipc_sections = list(ipc_data.keys())
# Enhanced text preparation with more context but without punishment details
ipc_texts = [
    f"{ipc} {ipc_data[ipc]['offense']} {ipc_data[ipc]['description']} {ipc_data[ipc]['simple_words']}"
    for ipc in ipc_sections
]

# Precompute punishment severity for each IPC section
ipc_punishment_severity = {}
for ipc in ipc_sections:
    punishment_text = ipc_data[ipc].get('punishment', '')
    ipc_punishment_severity[ipc] = estimate_punishment_severity(punishment_text)

# Preprocess all texts
ipc_texts_processed = [preprocess_text(text) for text in ipc_texts]

# -------------------------------
# LOAD OR CREATE EMBEDDINGS
# -------------------------------
embed_file_name = f"ipc_embeddings_{MODEL_NAME.replace('/', '_')}.pkl"

if os.path.exists(embed_file_name):
    print(f"Loading precomputed embeddings from {embed_file_name}...")
    with open(embed_file_name, "rb") as f:
        ipc_sections, ipc_embeddings = pickle.load(f)
else:
    print("enerating embeddings (first time, may take a while)...")
    ipc_embeddings = model.encode(ipc_texts_processed, convert_to_tensor=True)
    with open(embed_file_name, "wb") as f:
        pickle.dump((ipc_sections, ipc_embeddings), f)
    print(f"Embeddings saved to {embed_file_name}")

# -------------------------------
# LOAD OR CREATE TF-IDF VECTORS (for hybrid search)
# -------------------------------
if os.path.exists(TFIDF_FILE): 
    print(f" Loading precomputed TF-IDF vectors from {TFIDF_FILE}...")
    with open(TFIDF_FILE, "rb") as f:
        tfidf_vectorizer, tfidf_matrix = pickle.load(f)
else:
    print(" Generating TF-IDF vectors (first time)...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)  # Include unigrams and bigrams
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(ipc_texts_processed)
    with open(TFIDF_FILE, "wb") as f:
        pickle.dump((tfidf_vectorizer, tfidf_matrix), f)
    print(f"TF-IDF vectors saved to {TFIDF_FILE}")

# -------------------------------
# HYBRID SEARCH FUNCTION WITH SEVERITY CALIBRATION
# -------------------------------
def find_ipc_hybrid(user_input: str, top_k: int = 3):
    """Find IPC sections using hybrid semantic + keyword search with severity calibration"""
    # Preprocess user input
    processed_input = preprocess_text(user_input)

    # Estimate crime severity from user input
    crime_severity = estimate_crime_severity(user_input)

    # Semantic similarity
    query_embedding = model.encode(processed_input, convert_to_tensor=True)
    semantic_scores = util.pytorch_cos_sim(query_embedding, ipc_embeddings)[0].cpu().numpy()

    # Keyword similarity (TF-IDF)
    query_tfidf = tfidf_vectorizer.transform([processed_input])
    keyword_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    # Normalize scores
    semantic_scores = (semantic_scores - np.min(semantic_scores)) / (np.max(semantic_scores) - np.min(semantic_scores))
    keyword_scores = (keyword_scores - np.min(keyword_scores)) / (np.max(keyword_scores) - np.min(keyword_scores))

    # Combine scores
    hybrid_scores = (HYBRID_WEIGHT_EMBEDDING * semantic_scores +
                     HYBRID_WEIGHT_KEYWORD * keyword_scores)

    # Apply severity calibration
    calibrated_scores = []
    for i, score in enumerate(hybrid_scores):
        ipc = ipc_sections[i]
        punishment_severity = ipc_punishment_severity[ipc]

        # Penalize scores if punishment severity doesn't match crime severity
        severity_diff = abs(punishment_severity - crime_severity)
        severity_penalty = 1.0 - (severity_diff * 0.3)  # Reduce score by up to 30% for severity mismatch

        calibrated_score = score * severity_penalty
        calibrated_scores.append(calibrated_score)

    calibrated_scores = np.array(calibrated_scores)

    # Get top results
    top_indices = np.argsort(calibrated_scores)[::-1][:top_k*2]  # Get more initially to filter

    results = []
    for idx in top_indices:
        score = calibrated_scores[idx]

        # Apply threshold
        if score < SIMILARITY_THRESHOLD and len(results) >= top_k:
            continue

        ipc = ipc_sections[idx]
        details = ipc_data[ipc]

        results.append({
            "IPC": ipc,
            "offense": details["offense"],
            "punishment": details["punishment"],
            "cognizable": details["cognizable"],
            "bailable": details["bailable"],
            "court": details["court"],
            "score": float(score),
            "crime_severity": crime_severity,
            "punishment_severity": ipc_punishment_severity[ipc],
            "severity_match": 1.0 - abs(crime_severity - ipc_punishment_severity[ipc])
        })

        # Stop if we have enough results
        if len(results) >= top_k:
            break

    # Sort by final score
    results.sort(key=lambda x: x["score"], reverse=True)

    return results

# -------------------------------
# DISPLAY RESULTS
# -------------------------------
def display_results(matches, user_input):
    """Display search results in a formatted way"""
    print(f"\nTop Matches for: '{user_input}'\n" + "="*70)

    if not matches:
        print("No relevant IPC sections found. Try a more detailed description.")
        return

    for i, match in enumerate(matches, 1):
        print(f"{i}. Section: {match['IPC']} (Confidence: {match['score']:.4f})")
        print(f"    Offense: {match['offense']}")
        print(f"    Punishment: {match['punishment']}")
        print(f"    Cognizable: {match['cognizable']}")
        print(f"    Bailable: {match['bailable']}")
        print(f"    Court: {match['court']}")

        # Show severity information
        crime_sev = match['crime_severity']
        punish_sev = match['punishment_severity']
        severity_match = match['severity_match']

        print(f"   Crime Severity: {crime_sev:.2f}, Punishment Severity: {punish_sev:.2f}")
        print(f"   Severity Match: {severity_match:.2f}")

        if severity_match < 0.7:
            print("Note: Severity mismatch detected - this section might be too harsh")

        print("-"*70)

# -------------------------------
# EXAMPLE USAGE
# -------------------------------
if __name__ == "__main__":
    print("   IPC Code Classifier - Enhanced with Severity Calibration")
    print("   Enter a crime description to find relevant IPC sections")
    print("   Type 'quit' to exit\n")

    while True:
        user_query = input("Enter a crime description: ").strip()

        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_query:
            print("Please enter a description.")
            continue

        print("Analyzing and searching...")
        matches = find_ipc_hybrid(user_query, top_k=3)

        display_results(matches, user_query)

        # Show confidence level
        if matches:
            avg_confidence = sum(match['score'] for match in matches) / len(matches)
            if avg_confidence < 0.5:
                print("Low confidence results. Consider providing more details about the crime.")
            elif avg_confidence < 0.7:
                print("Medium confidence results. The matches might need verification.")
            else:
                print("High confidence results. These are likely accurate matches.")
