# 📘 Minimális RAG példa – Oktatási célú bevezető

Ez a projekt a **RAG (Retrieval-Augmented Generation)** legegyszerűbb, oktatási célú implementációját mutatja be **OpenAI** alapon.  
Célja, hogy a tanulók megértsék:

- hogyan működik a dokumentumfeldolgozás,
- mi az a chunking,
- hogyan készülnek az embeddingek,
- hogyan zajlik az egyszerű embedding-alapú keresés,
- és hogyan ad át kontextust a rendszer egy nyelvi modellnek.

A projekt minimális függőséggel, könnyen átlátható szerkezettel rendelkezik, így tökéletes kiindulópont a RAG alapjainak megértéséhez.

---

# 🚀 Előkészületek

## 1. Python ellenőrzése

Győződj meg róla, hogy **Python 3.10+** telepítve van:

```bash
python --version
```

---

## 2. Virtuális környezet létrehozása

A projekt gyökerében futtasd:

```bash
python -m venv .venv
```

### Aktiválás (Windows + VS Code)

```bash
.venv\Scripts\activate
```

Siker esetén a parancssor elején megjelenik:

```
(.venv)
```

---

## 3. Függőségek telepítése

```bash
pip install -r requirements.txt
```

---

## 4. OpenAI kulcs beállítása

A projekt gyökerében hozz létre egy `.env` fájlt, és add meg benne:

```
OPENAI_API_KEY=ide_ird_a_sajat_kulcsod
```

A repó tartalmaz egy `.env.sample` mintafájlt is.  
A `.env` fájlt *ne* verziókezeld.

---

# ▶️ A példa futtatása

```bash
python main.py
```

A program:

1. betölti a dokumentumot,  
2. chunkokra bontja (mondathatár-érzékeny módon),  
3. elkészíti az embeddingeket,  
4. cosine similarity alapján kiválasztja a legrelevánsabb chunkokat,  
5. a TOP‑K kontextust átadja az OpenAI modellnek,  
6. majd választ generál.

---

# 📂 Projekt struktúra

```
project_root/
│
├── README.md
├── requirements.txt
├── .env.sample
│
├── main.py
├── loader.py
├── embedder.py
├── searcher.py
│
└── data/
    └── example.txt
```

---

# 🎯 A projekt célja

Ez egy minimális, oktatási célú RAG-példa, amely bemutatja:

- dokumentum feldolgozását,
- darabolást (chunking),
- embedding készítést,
- cosine similarity alapú visszakeresést,
- kontextus alapú válaszgenerálást OpenAI modellel.

A projekt a RAG‑alapú alkalmazások megértésének **első, legegyszerűbb lépcsője**.
