#!/usr/bin/env python
# coding: utf-8

# In[35]:


pwd


# In[36]:


# Importing important libraries
import numpy as np 
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


# In[37]:


# Loading the data
df = pd.read_csv('train.csv') 
df.head()


# In[38]:


# Seperating the important columns 
df = df[['text', 'label']]
df.head()


# In[39]:


# Dropping the null values
df.dropna(inplace = True)


# In[40]:


# Splitting the data into training and testing 
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], 
                                                                    df['label'],
                                                                   test_size = 0.2, 
                                                                   random_state = 5)

# Loading the pre-trained BERT tokenizer 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[41]:


# Creating a function to tokenize 
def tokenize_function(texts, labels):
    encodings = tokenizer(texts.tolist(), truncation = True, padding = True, max_length = 512)
    return encodings, labels.tolist()


# In[ ]:


# Encoding the training and testing data
train_encodings, train_labels = tokenize_function(train_texts, train_labels)
val_encodings, val_labels = tokenize_function(val_texts, val_labels)


# In[20]:


# Creating a class 'NewsDataset'
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key : torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item 
        
    def __len__(self):
        return len(self.labels)


# In[21]:


# Getting the train dataset and testing dataset
train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)


# In[22]:


# Loading the pre-trained BERT Model
from transformers import logging
logging.set_verbosity_error()
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')


# In[23]:


# Set up training parameters
train_loader = DataLoader(train_dataset, batch_size = 8, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 8, shuffle = False)
optim = AdamW(model.parameters(), lr = 5e-5)


# In[24]:


# Training function 
def train(model, data_loader, optim):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optim.zer_grad()
        inputs = {key:val.to(device) for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)
        outputs = model(**inputs, labels = labels)
        loss = output.loss
        total_loss += loss.item()
        loss.backward()
        optim.step()
    return total_loss / len(data_loader)


# In[25]:


# Evaluation Function 
def evaluate(model, data_loader):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in data_laoder:
            inputs = {key: val.to(device) for ke, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            preds.extend(torch.argmax(outputs.logits, axis = 1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
        return metrics.accuracy_score(true_labels, preds), metrics.classification_report(true_labels, preds)


# In[26]:


# Training Loops 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in range(3):
    train_loss = train(model, train_loader, optim)
    val_accuracy, val_report = evaluate(model, val_loader)
    print(f'Epoch : {epoch+1}, Train Loss : {train_loss}, Validation Accircy : {val_accuracy}')
    print(val_report)


# In[ ]:


# Save the model 
model.save_pretrained('fake_news_detector')
tokenizer.save_pretrained('fake_news_detector')


# In[ ]:




