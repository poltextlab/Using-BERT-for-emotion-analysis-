# Using BERT for emotion analysis – A novel, resource constrained emotions classification approach with test on a Hungarian media corpus 

# Datasets and codes for the paper mentioned in the title

# 1. bert.py processes texts from etl_x.tsv
# 2. gridsearch_x.py fits 100 optimized models and saves results in easily parseable JSON-files

# a. etl_x.tsv files are prepared corpora for extracting BERT-embeddings
# b. featuresfinal_x.npy are numpy files with BERT-embeddings, labels_x.npy are corresponding labels
# c. v1x_corpus files are conventionally preprocessed, tf-idf weighted document-term matrices without the "document" column, v1x_labels files are corresponding labels
