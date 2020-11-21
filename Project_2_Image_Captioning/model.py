import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        ''' Initialize the layers of this model.'''
        super(DecoderRNN, self).__init__()
        
        #init the decoder
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.device = device
        #--------#
        # Layers #
        #--------#
        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(input_size= self.embed_size, 
                            hidden_size= self.hidden_size,
                            num_layers= self.num_layers,
                            batch_first= True,
                            bidirectional=False,
                            dropout = 0)

        
        # the linear layer that maps the hidden state output dimension 
        self.fc_1 = nn.Linear(self.hidden_size, self.vocab_size)
        

    def init_hidden(self, batch_size):
        # initialize the hidden state (see code below)
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.hidden_size, device = device),
                torch.zeros(1, batch_size, self.hidden_size, device = device))
        
    
    def forward(self, features, captions):
        
        # Initialize the hidden state
        self.hidden = self.init_hidden(features.shape[0])# features is of shape (batch_size, embed_size)
        
        ''' Define the feedforward behavior of the model.'''
        # create embedded word vectors for each word in a sentence
        embeddings = self.word_embeddings(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        lstm_out, self.hidden = self.lstm( inputs, self.hidden)
        return self.fc_1(lstm_out)
    
        # not necessary, it is included in CROSSENTROPYLOSS 
        #return self.fc_1(nn.functional.log_softmax(lstm_out, dim=1))
   
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        
        # Initialize the hidden state
        # features is of shape: batch-size embedding-size
        hidden = self.init_hidden(inputs.shape[0])
        
        # list of words
        words = []
        
        # get the predicted words
        with torch.no_grad():
            # loop until the end we exceed the max length
            while len(words) < max_len:
                
                # get the lstm values
                lstm_output, hidden = self.lstm(inputs, hidden)
                
                # predict the class of the words
                predictions = self.fc_1(lstm_output)
                predictions = predictions.squeeze(1)
                
                # take the word with the highest prob
                prediction = predictions.argmax(dim=1)
                
                words.append(prediction.item())
                
                # prepare the next input
                inputs = self.word_embeddings(prediction.unsqueeze(0))
                
                # stop, if the prediction has vale 1 --> end
                if prediction == 1:
                    break
        
        return words