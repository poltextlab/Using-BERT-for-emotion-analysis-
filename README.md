# Using BERT for emotion analysis – A novel, resource constrained emotions classification approach with test on a Hungarian media corpus 

Datasets and codes for the paper mentioned in the title

**bert.py** processes texts from etl_x.tsv, **gridsearch_x.py** fits 100 optimized models and saves results in easily parseable JSON-files

Files included in the repo:
1. etl_x.tsv files are prepared corpora for extracting BERT-embeddings
2. featuresfinal_x.npy are numpy files with BERT-embeddings, labels_x.npy are corresponding labels
3. v1x_corpus files are conventionally preprocessed, tf-idf weighted document-term matrices without the "document" column, v1x_labels files are corresponding labels
