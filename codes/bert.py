
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from transformers import AutoTokenizer, AutoModel

import pandas as pd
import numpy as np
import os

#INPUT = 'etl_nothree'
INPUT='test'

#DIVISOR = 200
DIVISOR = 3

# different setups might need different solutions here
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc')
model = AutoModel.from_pretrained('SZTAKI-HLT/hubert-base-cc')

# read corpus from tsv
corpus = pd.read_csv(f"corpora_and_labels/{INPUT}.tsv", sep='\t')
print(corpus)

# tokenize corpus
# note: it does not check for length requirements at this time
tokenized = corpus["text"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
print(tokenized)

# create padding and attention masks
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
print(padded)

attention_mask = np.where(padded != 0, 1, 0)
print(attention_mask)

# for computationally weaker setups, batch execution is the only way to process the texts
# manipulate floor divisor if a different batch size is needed
batchsize = (len(corpus) // DIVISOR) + 1
print('Number of batches:', batchsize)
splitpadded = np.array_split(padded, batchsize)
splitmask = np.array_split(attention_mask, batchsize)


last_hidden_states = []
model = model.to(device)
featuresfinal = np.empty((0, 768), dtype='float32')

# take batches of tokenized texts
# to extract BERT's last hidden states, i.e. contextual word embeddings
#
# XXX handling attention_mask was erroneous here,
# because array_split() gives variable length!
# now: zip() ensures that text and attention data is taken strictly in parallel
for count, (batch, mask) in enumerate(zip(splitpadded, splitmask)):
    batch_cnt = count + 1
    print(f'Batch #{batch_cnt}')
    paddedsplit = np.array(batch, dtype='float64')

    input_batch = torch.tensor(batch).to(torch.long)
    mask_batch = torch.tensor(mask)
    print(input_batch)
    print(mask_batch)
    print('Batches established!')

    # put data onto GPU
    input_batch = input_batch.to(device)
    mask_batch = mask_batch.to(device)
    print('Lengths', input_batch.size(0), mask_batch.size(0))

    # no_grad ensures there is no gradient update in the model,
    # as we are not looking for recursive training here
    with torch.no_grad():
        print('Model is running on', model.device)
        last_hidden_states = model(input_batch, attention_mask=mask_batch)
    print('Hidden states created for batch', batch_cnt)

    features = last_hidden_states[0][:, 0, :].cpu().numpy()
    featuresfinal = np.append(featuresfinal, features, axis=0)
    print('Finished with batch', batch_cnt)

# output + labels are saved as separate files
labels = corpus["topik"]

np.save(f"featuresfinal_{INPUT}", featuresfinal)
np.save(f"labels_{INPUT}", labels)

print(list(featuresfinal))
print(labels)

