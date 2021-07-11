import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import os

# the first section is custom made, different setups might need different solutions here
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_precorpused('SZTAKI-HLT/hubert-base-cc')
model = AutoModel.from_precorpused('SZTAKI-HLT/hubert-base-cc')

# the corpus is read from a tsv, tokenized (note that it does not check for length requirements at this time)
corpus = pd.read_csv("etl.tsv", sep='\t')
tokenized = corpus["text"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# the following section creates padding and attention masks
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
attention_mask = np.where(padded != 0, 1, 0)

# for computationally weaker setups, batch execution is the only way to process the texts
# the floor divisor is the variable to manipulate here if a difference in batch size is needed
sectionsize = (len(corpus) // 200) + 1
print('Section number:', sectionsize)
splitpadded = np.array_split(padded, sectionsize)


last_hidden_states = []
model = model.to(device)
featuresfinal = np.empty((0, 768), dtype='float32')

# last_hidden function takes in batches of tokenized texts to extract BERT's last hidden states, i.e. contextual word embeddings
def last_hidden():
    for count, i in enumerate(splitpadded):
        paddedsplit = np.array(i, dtype='float64')
        length = len(paddedsplit)
        #continuous feedback is received from the runs, to ensure the process is not hung and the model is running on gpu
        print('Length established!')
        input_batch = torch.tensor(i).to(torch.long)
        mask_batch = torch.tensor(attention_mask[length*count:length*count+length])
        print('Batches established!')
        input_batch = input_batch.to(device)
        mask_batch = mask_batch.to(device)
        print('Lengths', length, input_batch.size(0), mask_batch.size(0))
        # no_grad ensures there is no gradient update in the model, as we are not looking for recursive training here
        with torch.no_grad():
            print('Model is running on', model.device)
            global last_hidden_states
            last_hidden_states = model(input_batch, attention_mask=mask_batch)
        print('Hidden states created for batch', count+1)
        global features
        features = last_hidden_states[0][:, 0, :].cpu().numpy()
        global featuresfinal
        featuresfinal = np.append(featuresfinal, features, axis=0)
        print('Finished with batch', count+1)

# the output of the function and the labels are saved as separate files, so this script can be abandoned
last_hidden()
np.save("featuresfinal", featuresfinal)
labels = corpus["topik"]
np.save("labels", labels)
