#file: main.py
"""
# Egyszerű RAG példa – belépési pont
# A modul végigviszi a teljes RAG folyamatot:
# 1. Dokumentum betöltése
# 2. Embeddingek készítése
# 3. TOP-K keresés cosine similarity alapján
# 4. Kontextus átadása az OpenAI modellnek
"""

from loader import load_text_document
from embedder import generate_embeddings
from searcher import search_similar_chunk
from openai import OpenAI
import os


# --- OpenAI kliens inicializálása ---
def create_openai_client():
    """Létrehozza az OpenAI kliens objektumot a környezeti változó alapján."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("HIBA: Nincs beállítva az OPENAI_API_KEY környezeti változó!")
    return OpenAI(api_key=api_key)


# --- Fő futtatási logika ---
def run_rag_pipeline():
    """A teljes RAG folyamat futtatása: betöltés → embedding → keresés → válasz generálása."""

    print("Előkészítés folyamatban... (dokumentum betöltése és beágyazása)")

    # Dokumentum betöltése
    document_text = load_text_document("data/example.txt")

    # Embeddingek létrehozása
    chunks, chunk_embeddings = generate_embeddings(document_text)

    print("Előkészítés kész.\n")

    # Felhasználói kérdés bekérése
    user_question = input("Adj meg egy kérdést a dokumentummal kapcsolatban: ")

    # TOP-K chunkok similarity értékekkel
    best_chunks_with_scores = search_similar_chunk(user_question, chunks, chunk_embeddings)

    print("\n--- Legrelevánsabb találatok ---")

    for i, (chunk, score) in enumerate(best_chunks_with_scores, start=1):
        print(f"\n{i}. részlet (similarity: {score:.4f})")
        print(chunk)

    # A modell számára összefűzzük a TOP-K chunkot egyetlen kontextussá
    combined_context = "\n\n".join(chunk for chunk, _ in best_chunks_with_scores)

    # OpenAI kliens
    client = create_openai_client()

    # Válasz generálása a felhasználói kérdés + összefűzött kontextus alapján
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Használd a megadott kontextust a válaszhoz."},
            {
                "role": "user",
                "content": (
                    f"Kérdés: {user_question}\n\n"
                    f"Kontextus (TOP-K találatok):\n{combined_context}"
                ),
            },
        ],
    )

    print("\n--- Modell válasza ---")
    print(response.choices[0].message.content)


# --- Futtatás ---
if __name__ == "__main__":
    run_rag_pipeline()
