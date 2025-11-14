#file: loader.py
"""
# Dokumentumbetöltő modul
# Ez a modul felelős egy egyszerű szöveges állomány tartalmának beolvasásáért.
# Oktatási cél: megmutatni, hogyan válik a RAG folyamat részévé a bemeneti adat.
"""

import os


def load_text_document(file_path: str) -> str:
    """Beolvas egy szöveges dokumentumot és visszaadja a tartalmát."""

    # Ellenőrizzük, hogy létezik-e a fájl
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HIBA: A dokumentum nem található: {file_path}")

    # Fájl beolvasása UTF-8 formátumban
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    return content
