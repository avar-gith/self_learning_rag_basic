#file: embedder.py
"""
# Embedding készítő modul
# Ez a modul feldolgozza a dokumentum szövegét, kisebb részekre bontja (chunking),
# majd OpenAI embedding modellt használva vektort hoz létre mindegyik részlethez.
# Cél: megmutatni, hogyan alakítjuk a nyers szöveget később kereshető vektorrá.
"""

from openai import OpenAI
import os


# --- OpenAI kliens létrehozása embeddinghez ---
def create_embedding_client() -> OpenAI:
    """Visszaad egy OpenAI klienst az embedding generáláshoz."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("HIBA: Nincs beállítva az OPENAI_API_KEY környezeti változó!")

    return OpenAI(api_key=api_key)


# --- Mondathatár-érzékeny chunkolás ---
def chunk_text(document_text: str, max_length: int = 300) -> list[str]:
    """Felbontja a szöveget kisebb részekre, lehetőség szerint mondathatáron vágva.

    Működés:
    - A chunk hossza max. `max_length` karakter. (fenti int = 300 a példában)
    - A függvény visszafelé keres egy mondatvégi írásjelet (., !, ?).
    - Ha talál, és nem túl messze van a chunk végétől → ott vág.
    - Ha nincs mondathatár a tartományban → egyszerűen max_length szerint vágunk.

    Ez a megközelítés robusztusabbá teszi az embeddingeket, mert nem szakadnak ketté a mondatok.
    A Példában átfedést nem alkalmazunk, de egy minimális bemutató kód alább megtalálható.
    """

    chunks = []
    start = 0

    while start < len(document_text):
        # Alap vágási pont
        end = min(start + max_length, len(document_text))
        candidate = document_text[start:end]

        # Mondathatár keresése visszafelé
        sentence_end = max(
            candidate.rfind("."),
            candidate.rfind("!"),
            candidate.rfind("?")
        )

        # Ha található mondatvég, és a chunk legalább 30%-án túl van
        if sentence_end != -1 and sentence_end > len(candidate) * 0.3:
            # Ott vágunk
            end = start + sentence_end + 1
            chunk = document_text[start:end].strip()
        else:
            # Nincs mondathatár a tartományban → sima vágás
            chunk = candidate.strip()

        chunks.append(chunk)
        start = end

    return chunks


# --- Egyszerűbb további példák (token alapú chunkolás) ---

# Megjegyzés példa token alapú chunkolásra:
# További csomagokat igényel pl. tiktoken
#
# 1. Token alapú chunkolás egyszerűen:
#    tokenizer = tiktoken.get_encoding("cl100k_base")
#    tokens = tokenizer.encode(document_text)
#    for i in range(0, len(tokens), 200):
#        chunk = tokenizer.decode(tokens[i:i+200])
#
# 2. Token alapú chunkolás átfedéssel:
#    chunk_size = 200
#    overlap = 50
#    for i in range(0, len(tokens), chunk_size - overlap):
#        chunk = tokenizer.decode(tokens[i:i+chunk_size])


# --- Embedding generálás ---
def generate_embeddings(document_text: str):
    """Chunkokra bontja a szöveget, majd embeddinget készít minden részlethez.

    Visszatérés:
    - chunks: a feldarabolt szövegrészek listája
    - embeddings: a hozzájuk tartozó vektorok listája
    """

    client = create_embedding_client()

    # Szöveg chunkolása
    chunks = chunk_text(document_text)

    embeddings = []

    for chunk in chunks:
        # Embedding generálása OpenAI modellel
        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )

        embedding_vector = embedding_response.data[0].embedding
        embeddings.append(embedding_vector)

    return chunks, embeddings
