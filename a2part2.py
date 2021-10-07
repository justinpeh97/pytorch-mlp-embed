import os
import re
import sys
import string
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


torch.manual_seed(0)


class LangDataset(Dataset):
    """
    Define a pytorch dataset class that accepts a text path, and optionally label path and
    a vocabulary (depends on your implementation). This class holds all the data and implement
    a __getitem__ method to be used by a Python generator object or other classes that need it.

    DO NOT shuffle the dataset here, and DO NOT pad the tensor here.
    """
    def __init__(self, text_path, label_path=None, vocab=None):
        """
        Read the content of vocab and text_file
        Args:
            text_path (string): Path to the text file.
            label_path (string, optional): Path to the label file.
            vocab (string, optional): You may use or not use this
        """
        with open(text_path) as f:
            self.texts = f.read().splitlines()
        
        self.bigrams = []
        if vocab == None:
          index = 1
          
          self.unique_bi = {} 
          for line in self.texts:
              for i in range(len(line)-1):
                  if line[i:i+2] not in self.unique_bi.keys():
                      self.unique_bi[line[i:i+2]] = index
                      index += 1
        else:
          self.unique_bi = vocab


                
        for line in self.texts:
            bigrams = []
            for i in range(len(line)-1):
                bigram = line[i:i+2]
                if bigram in self.unique_bi.keys():
                  bigrams.append(self.unique_bi[line[i:i+2]])
                else:
                  bigrams.append(0)
            self.bigrams.append(bigrams)

        self.label_path = label_path
        if label_path != None:
            
            with open(label_path) as f:
                self.labels = f.read().splitlines()
            self.unique_labels = []
            for label in self.labels:
                if label not in self.unique_labels:
                    self.unique_labels.append(label)
                
    def vocab_size(self):
        """
        A function to inform the vocab size. The function returns two numbers:
            num_vocab: size of the vocabulary
            num_class: number of class labels
        """

        return len(self.unique_bi.keys()) + 1, len(self.unique_labels)
    
    def __len__(self):
        """
        Return the number of instances in the data
        """
        return len(self.texts)

    def __getitem__(self, i):
        """
        Return the i-th instance in the format of:
            (text, label)
        Text and label should be encoded according to the vocab (word_id).
        both text and label are recommended to be in pytorch tensor type.
        
        DO NOT pad the tensor here, do it at the collator function.
        """
        
        if self.label_path == None:
            return torch.tensor(self.bigrams[i]), torch.tensor(0)
        else:
            label = self.labels[i]
            label_index = self.unique_labels.index(label)     
            return torch.tensor(self.bigrams[i]), torch.tensor(label_index)


def mean_nopad(x):
    mask = x != 0
    return (x*mask).sum(dim=1)/mask.sum(dim=1)
    
class Model(nn.Module):
    """
    Define a model that with one embedding layer, a hidden
    feed-forward layer, a dropout layer, and a feed-forward
    layer that reduces the dimension to num_class
    """
    def __init__(self, num_vocab, num_class, dropout=0.3):
        super().__init__()
        w_dim = 50
        size_layer = 100
        self.embeddings = torch.nn.Embedding(num_vocab, w_dim, padding_idx = 0)
        self.linear1 = torch.nn.Linear(w_dim, size_layer)
        self.linear2 = torch.nn.Linear(size_layer, num_class)
        self.ReLU = torch.nn.ReLU() 
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax()

    def forward(self, x):
        
        h0 = self.embeddings(x.long())
        h0 = self.dropout(h0)
        h0 = mean_nopad(h0)
        h1 = self.linear1(h0)
        h1 = self.ReLU(h1)
        h1 = self.dropout(h1)
        h2 = self.linear2(h1)

        return h2


def collator(batch):

    """
    Define a function that receives a list of (text, label) pair
    and return a pair of tensors:
        texts: a tensor that combines all the text in the mini-batch, pad with 0
        labels: a tensor that combines all the labels in the mini-batch
    """
    texts, labels = zip(*batch)
    max_len = max([row.size()[0] for row in texts])
    features = torch.zeros((len(texts), max_len))
    for i in range(len(texts)):
        features[i, 0:texts[i].size()[0]] = texts[i]
    labels = torch.tensor([sample[1] for sample in batch] )
    
    return features, labels


def train(model, dataset, batch_size, learning_rate, num_epoch, device='cpu', model_path=None):
    """
    Complete the training procedure below by specifying the loss function
    and the optimizer with the specified learning rate and specified number of epoch.
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)

    # assign these variables
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(data_loader, 0):
            # get the inputs; data is a tuple of (inputs, labels
            texts = data[0].to(device)
            labels = data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # do forward propagation
            yhat = model(texts)
            # do loss calculation
            loss = criterion(yhat, labels)
            # do backward propagation
            loss.backward()
            # do parameter optimization step
            optimizer.step()
            # calculate running loss value for non padding
            running_loss += loss
            # print loss value every 100 steps and reset the running loss
            if step % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, running_loss / 100))
                running_loss = 0.0

    end = datetime.datetime.now()
    
    # save the model weight in the checkpoint variable
    # and dump it to system on the model_path
    # tip: the checkpoint can contain more than just the model
    torch.save({'epoch':epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':loss,
                'vocab': dataset.unique_bi,
                'num_vocab_num_class': dataset.vocab_size()},
               model_path)

    print('Model saved in ', model_path)
    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))


def test(model, dataset, class_map, device='cpu'):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=20, collate_fn=collator, shuffle=False)
    labels = []
    with torch.no_grad():
        for data in data_loader:
            texts = data[0].to(device)
            outputs = model(texts).cpu()
            # get the label predictions
            for output in outputs:
                label = class_map[int(torch.argmax(output))]                
                labels.append(label)
    return labels


def main(args):
    if torch.cuda.is_available():
        device_str = 'cuda:{}'.format(0)
    else:
        device_str = 'cpu'
    device = torch.device(device_str)
    
    assert args.train or args.test, "Please specify --train or --test"
    if args.train:
        assert args.label_path is not None, "Please provide the labels for training using --label_path argument"
        train_dataset = LangDataset(args.text_path, args.label_path)
        num_vocab, num_class = train_dataset.vocab_size()
        model = Model(num_vocab, num_class).to(device)
        
        # you may change these hyper-parameters
        learning_rate = 0.001
        batch_size = 20
        num_epochs = 50

        train(model, train_dataset, batch_size, learning_rate, num_epochs, device, args.model_path)
    if args.test:
        assert args.model_path is not None, "Please provide the model to test using --model_path argument"
        # the lang map should map the class index to the language id (e.g. eng, fra, etc.)
        lang_map = {0:'eng', 1:'deu', 2:'fra', 3:'ita', 4:'spa'}

        # initialize and load the model
        checkpoint = torch.load(args.model_path)
        vocab_output = checkpoint['vocab']
        num_vocab, num_class = checkpoint['num_vocab_num_class']
        model = Model(num_vocab, num_class).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # create the test dataset object using LangDataset class
        dataset = LangDataset(args.text_path, None, vocab = vocab_output)

        # run the prediction
        preds = test(model, dataset, lang_map, device)
        
        # write the output
        with open(args.output_path, 'w', encoding='utf-8') as out:
            out.write('\n'.join(preds))
    print('\n==== A2 Part 2 Done ====')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_path', help='path to the text file')
    parser.add_argument('--label_path', default=None, help='path to the label file')
    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--test', default=False, action='store_true', help='test the model')
    parser.add_argument('--model_path', required=True, help='path to the output file during testing')
    parser.add_argument('--output_path', default='out.txt', help='path to the output file during testing')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)
