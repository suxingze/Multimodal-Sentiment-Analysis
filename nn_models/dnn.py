from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal

class DNN(nn.Module):
    '''
    The subnetwork that is used in TFN in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(DNN, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3 
    
class Fusion(nn.Module):
    def __init__(self, input_sizes, hidden_sizes, dropouts, post_fusion_dim, output_size, fusion_type='tfn'):
        '''
        Args:
            input_sizes - a length-3 list, contains [text_dim, visual_dim, audio_dim]
            hidden_sizes - another length-3 list, similar to input_sizes
            dropouts - a length-4 list, contains [text_dropout, visual_dropout, audio_dropout, post_fusion_dropout]
            post_fusion_dim - int, specifying the size of the sub-networks after tensorfusion
            output_size - int, specifying the number of classes. Here a continuous prediction is produced which will be used for binary classification
            fusion_type - 'concat': simple concatenation and fully connected; 'tfn': tensor fusion network fusion.
        '''
        super(Fusion, self).__init__()
        if not (isinstance(input_sizes, (list, tuple)) and len(input_sizes) == 3):
            raise ValueError(f"input_sizes must be a list or tuple of length 3, got {input_sizes}")
        if not (isinstance(hidden_sizes, (list, tuple)) and len(hidden_sizes) == 3):
            raise ValueError(f"hidden_sizes must be a list or tuple of length 3, got {hidden_sizes}")
        if not (isinstance(dropouts, (list, tuple)) and len(dropouts) == 4):
            raise ValueError(f"dropouts must be a list or tuple of length 4, got {dropouts}")

        self.fusion_type = fusion_type

        # dimensions are specified
        self.text_in = input_sizes[0]
        self.video_in = input_sizes[1]
        self.audio_in = input_sizes[2]

        self.text_hidden = hidden_sizes[0]
        self.video_hidden = hidden_sizes[1]
        self.audio_hidden = hidden_sizes[2]
        self.post_fusion_dim = post_fusion_dim
        self.post_fusion_prob = dropouts[3]

        self.output_size = output_size

        # define sub-networks
        self.text_dropout = dropouts[0]
        self.text_subnet = DNN(self.text_in, self.text_hidden, self.text_dropout)

        self.video_dropout = dropouts[1]
        self.video_subnet = DNN(self.video_in, self.video_hidden, self.video_dropout)

        self.audio_dropout = dropouts[2]
        self.audio_subnet = DNN(self.audio_in, self.audio_hidden, self.audio_dropout)

        self.text_out_dim = self.text_hidden
        self.video_out_dim = self.video_hidden
        self.audio_out_dim = self.audio_hidden

        # define the post fusion layers
        if self.fusion_type == 'simple':

            self.fc1 = nn.Linear(self.text_out_dim + self.video_out_dim + self.audio_out_dim, self.post_fusion_dim)
            self.fc2 = nn.Linear(self.post_fusion_dim, output_size)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(self.post_fusion_prob)
            self.bn = nn.BatchNorm1d(self.text_out_dim + self.video_out_dim + self.audio_out_dim)

        elif self.fusion_type == 'tfn':
            self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)

            self.post_fusion_layer_1 = nn.Linear((self.text_out_dim + 1) * (self.video_out_dim + 1) * (self.audio_out_dim + 1), self.post_fusion_dim)
            self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
            self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, self.output_size)

        else:
            raise ValueError('fusion_type can only be concat or tfn!')


    def forward(self, audio_x, video_x, text_x, lengths_x):
        '''
        Args:
            audio_x - a tensor with shape (max_len, batch_size, in)
            video_x - same as audio_x
            text_x - same as audio_x
        '''
        audio_x = torch.mean(audio_x, dim=0, keepdim=True).squeeze()  # (batch_size, in_dim)
        audio_h = self.audio_subnet(audio_x)

        video_x = torch.mean(video_x, dim=0, keepdim=True).squeeze()
        video_h = self.video_subnet(video_x)

        text_x = torch.mean(text_x, dim=0, keepdim=True).squeeze()
        text_h = self.text_subnet(text_x)

        batch_size = lengths_x.size(0) # lengths_x is a tensor of shape (batch_size,) 

        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor
        
        # in TFN we are doing a regression with constrained output range: (-3, 3), hence we'll apply sigmoid to output
        # shrink it to (0, 1), and scale\shift it back to range (-3, 3)
        self.output_range = torch.FloatTensor([6]).type(DTYPE)
        self.output_shift = torch.FloatTensor([-3]).type(DTYPE)

        if self.fusion_type == 'concat':
            h = torch.cat((text_h, video_h, audio_h), dim=1)
            h = self.bn(h)
            h = self.fc1(h)
            h = self.dropout(h)
            h = self.relu(h)
            o = self.fc2(h)
            return o

        elif self.fusion_type == 'tfn':
           # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
            _audio_h = torch.cat((torch.ones((batch_size, 1), requires_grad=False).type(DTYPE), audio_h), dim=1)
            _video_h = torch.cat((torch.ones((batch_size, 1), requires_grad=False).type(DTYPE), video_h), dim=1)
            _text_h = torch.cat((torch.ones((batch_size, 1), requires_grad=False).type(DTYPE), text_h), dim=1)

            # _audio_h has shape (batch_size, audio_in + 1), _video_h has shape (batch_size, _video_in + 1)
            # we want to perform outer product between the two batch, hence we unsqueenze them to get
            # (batch_size, audio_in + 1, 1) X (batch_size, 1, video_in + 1)
            # fusion_tensor will have shape (batch_size, audio_in + 1, video_in + 1)
            fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))
            
            # next we do kronecker product between fusion_tensor and _text_h. This is even trickier
            # we have to reshape the fusion tensor during the computation
            # in the end we don't keep the 3-D tensor, instead we flatten it
            fusion_tensor = fusion_tensor.view(-1, (self.audio_out_dim + 1) * (self.video_out_dim + 1), 1) 
            fusion_tensor = torch.bmm(fusion_tensor, _text_h.unsqueeze(1)).view(batch_size, -1)
    
            post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
            post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped))
            post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1))
            post_fusion_y_3 = F.sigmoid(self.post_fusion_layer_3(post_fusion_y_2))
            o = post_fusion_y_3 * self.output_range + self.output_shift
            return o

        else:
            raise ValueError('Fusion_type can only be specified as concat or tfn!')