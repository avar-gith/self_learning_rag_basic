Minimális RAG példa – Útmutató

Ez a projekt a RAG (Retrieval-Augmented Generation) legegyszerűbb, oktatási célú példáját mutatja be OpenAI alapon. A cél, hogy a tanulók gyorsan megértsék a RAG működési elvét, és minimális környezettel el tudják készíteni az első működő implementációt.

🚀 Előkészületek
1. Python ellenőrzése

Győződj meg róla, hogy Python 3.10+ telepítve van:

python --version
🧱 Virtuális környezet létrehozása

A projekt gyökérkönyvtárában futtasd:

python -m venv .venv
Aktiválás (Windows – VS Code használata mellett)
.venv\Scripts\activate

Siker esetén a parancssor elején megjelenik:

(.venv)
📦 Függőségek telepítése
pip install -r requirements.txt
🔑 OpenAI kulcs beállítása

Hozz létre egy .env fájlt a projekt gyökerében, és add meg benne a kulcsot:

OPENAI_API_KEY=ide_ird_a_sajat_kulcsod

(Ha nem .env fájlt szeretnél használni, a kulcsot beírhatod környezeti változóba is.)

▶️ A példa futtatása
python main.py

Ha minden jól ment, a script betölti a dokumentumot, elvégzi az embeddinget, majd egy egyszerű keresést követően választ generál.

📘 Fájlstruktúra (a legegyszerűbb RAG példához)
project_root/
│
├── README.md
├── requirements.txt
├── main.py
├── loader.py
├── embedder.py
├── searcher.py
└── data/
    └── example.txt
🎯 A projekt célja

Ez egy nagyon egyszerű, oktatási célú demonstráció:

egy dokumentum feldolgozása

darabolás (chunking)

embedding létrehozása

egyszerű cosine similarity alapú visszakeresés

kontextus átadása egy OpenAI hívásnak

A második szenárióban majd ennél tovább lépünk egy lokális embedding + Elasticsearch megoldás felé.

Ha szeretnéd, most elkészíthetjük a fájlstruktúra további elemeit is (loader.py, embedder.py, searcher.py, main.py).