# Using BERT for emotion analysis – A novel, resource constrained emotions classification approach with test on a Hungarian media corpus 

Datasets and codes for the paper mentioned in the title

`bert.py` processes texts from `etl_*.tsv`

`gridsearch_*.py` fits 100 optimized models and saves results in easily parseable JSON-files

Files included in the repo:
1. `etl_*.tsv` files are prepared corpora for extracting BERT-embeddings
2. `featuresfinal_*.npy` are numpy files with BERT-embeddings, `labels_*.npy` are corresponding labels
3. `v1[023]_corpus.tsv` files are conventionally preprocessed, tf-idf weighted document-term matrices without the "document" column, `v1[023]_labels.tsv` files are corresponding labels
