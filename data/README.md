# Corpus

The source PDF is **not committed** — it is ~16 MB and does not belong in git history.

## What to put here

Any medical reference PDF. This project was built and tested against the
*Gale Encyclopedia of Medicine* (Vol. 1), a public medical reference text.

Drop one or more `.pdf` files into this directory:

```
data/
└── your-medical-reference.pdf
```

`data/*.pdf` is gitignored, so nothing you add here will be committed.

## Then build the index

```bash
python store_index.py
```

This loads every PDF in `data/`, splits it into 500-character chunks with
20 characters of overlap, embeds each chunk with `all-MiniLM-L6-v2`, and
writes the vectors to your Pinecone index.

Run it once per corpus change. It is not idempotent — re-running appends.
