# Corpus

No corpus ships with this repository. Bring a medical reference PDF you are licensed to use.

## What to put here

This project was built against a commercial medical encyclopedia, which is copyrighted and is
therefore not redistributed here. Public-domain alternatives work just as well for trying it out —
for example, patient-education material from the U.S. National Library of Medicine (MedlinePlus)
or other U.S. government health publications.

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

This loads every PDF in `data/`, splits it into 500-character chunks with 20 characters of
overlap, embeds each chunk with `all-MiniLM-L6-v2` (384 dimensions), creates the Pinecone index if
it does not exist, and writes the vectors with each chunk's source file and page number.

Run it once per corpus change. It is not idempotent — re-running appends.
