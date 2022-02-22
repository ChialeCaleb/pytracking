import torch
import torch.nn as nn
from ltr.models.layers.blocks import conv_block
from ltr.models.kys.conv_gru import ConvGRUCell 

class EventGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, event_length):
        super().__init__()
        
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.event_length=event_length
        for i in range(self.event_length):
            exec('self.gru{} = ConvGRUCell(input_dim=self.input_dim,hidden_dim=self.hidden_dim, kernel_size=3)'.format(i))

    def forward(self, state_cur, event):
        event_list=[]
        event_list = torch.chunk(event,self.event_length,dim=1)
        for i in range(self.event_length):
            exec('state_cur = self.gru{}(event_list[0].squeeze(1),state_cur)'.format(i))
        state_next=state_cur
        return state_next