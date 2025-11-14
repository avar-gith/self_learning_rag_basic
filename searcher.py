#file: searcher.py
"""
# Kereső modul – cosine similarity alapú kiválasztás
# Ez a modul megkeresi a felhasználói kérdéshez legjobban illeszkedő szövegrészletet.
# Oktatási cél: megmutatni, hogy az embeddingek segítségével numerikus módon
# mérhető a hasonlóság a szöveg és a kérdés között.
"""

from openai import OpenAI
import os
import math


# --- Konfigurációs értékek ---
TOP_K = 3          # Hány legjobb chunkot adjon vissza
THRESHOLD = 0.4    # Minimális similarity érték (0–1 között)


# --- OpenAI kliens a kérdés embeddingjéhez ---
def create_embedding_client() -> OpenAI:
    """Visszaad egy OpenAI klienst a kereséshez szükséges embedding generáláshoz."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("HIBA: Nincs beállítva az OPENAI_API_KEY környezeti változó!")

    return OpenAI(api_key=api_key)


# --- Cosine similarity számítása ---
def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Kiszámítja két vektor cosine similarity értékét.

    Ez megmutatja, mennyire "egy irányba mutat" a két embedding.
    1.0 → teljesen hasonló, 0 → nincs kapcsolat.
    """

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# --- Hasonló chunk keresése (TOP-K + threshold) ---
def search_similar_chunk(
    user_question: str,
    chunks: list[str],
    chunk_embeddings: list[list[float]]
) -> list[tuple[str, float]]:
    """A kérdés embeddingjét elkészíti, majd a legjobb TOP_K chunkot adja vissza similarity értékkel együtt."""

    client = create_embedding_client()

    # Felhasználói kérdés embeddingje
    question_embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=user_question
    )
    question_embedding = question_embedding_response.data[0].embedding

    # Minden chunk similarity értékének kiszámítása
    scored_chunks = []
    for chunk, emb in zip(chunks, chunk_embeddings):
        score = cosine_similarity(question_embedding, emb)
        scored_chunks.append((chunk, score))

    # Csökkenő sorrend similarity alapján
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # THRESHOLD szűrés
    filtered = [(chunk, score) for chunk, score in scored_chunks if score >= THRESHOLD]

    # TOP-K vágás
    best = filtered[:TOP_K]

    # Ha minden kiesett → adjunk üres listát (nincs releváns találat)
    return best
