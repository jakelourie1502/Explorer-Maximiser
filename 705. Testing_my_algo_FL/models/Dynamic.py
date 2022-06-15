import sys


sys.path.append(".")
import torch
import torch.nn.functional as TF 
from config import Config
from models.res_block import resBlock
class Dynamic(torch.nn.Module):
    """
    Input: 
      state
    Notes:
     
    Outputs: 
      state

    """
    def __init__(self, device):
        self.cfg = Config()
        self.device = device
        self.state_size = self.cfg.state_size
        self.action_size = self.cfg.actions_size
        self.state_channels = self.cfg.model.state_channels
        self.actions_size = self.cfg.actions_size
        self.env_size = self.cfg.observable_size
        self.res_blocks = self.cfg.dynamic.res_blocks
        

        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = self.state_channels + 1, out_channels = self.state_channels, 
                                                                    kernel_size= self.cfg.dynamic.conv1['kernel_size'],
                                                                    stride = self.cfg.dynamic.conv1['stride'],
                                                                    padding = self.cfg.dynamic.conv1['padding'],
                                                                    # padding_mode='replicate'
                                                                    )
        self.bn1 = torch.nn.BatchNorm2d(self.state_channels)
        # self.conv2 = torch.nn.Conv2d(in_channels = self.state_channels, out_channels = self.state_channels, 
        #                                                             kernel_size= 1,
        #                                                             stride = 1,
        #                                                             padding = 0)
        # self.bn2 = torch.nn.BatchNorm2d(self.state_channels)
        self.resBlocks = torch.nn.ModuleList([resBlock(x, self.device) for x in self.res_blocks])
        #reward
        self.conv1x1_reward = torch.nn.Conv2d(in_channels=self.state_channels, out_channels=self.cfg.dynamic.reward_conv_channels, kernel_size=1,padding=0, stride=1)
        self.bn3 = torch.nn.BatchNorm2d(self.cfg.dynamic.reward_conv_channels)
        self.FC1 = torch.nn.Linear(self.state_size[0]*self.state_size[1]*self.cfg.dynamic.reward_conv_channels, self.cfg.dynamic.reward_hidden_dim)
        self.bn4 = torch.nn.BatchNorm1d(self.cfg.dynamic.reward_hidden_dim)
        self.FC2 = torch.nn.Linear(self.cfg.dynamic.reward_hidden_dim, self.cfg.dynamic.reward_support[2])
        
        #terminal
        self.conv1x1_terminal = torch.nn.Conv2d(in_channels=self.state_channels, out_channels=self.cfg.dynamic.terminal_conv_channels, kernel_size=1,padding=0, stride=1)
        self.bn5 = torch.nn.BatchNorm2d(self.cfg.dynamic.terminal_conv_channels)
        self.FC1t = torch.nn.Linear(self.state_size[0]*self.state_size[1]*self.cfg.dynamic.terminal_conv_channels, self.cfg.dynamic.terminal_hidden_dim)
        self.bn6 = torch.nn.BatchNorm1d(self.cfg.dynamic.terminal_hidden_dim)
        self.FC2t = torch.nn.Linear(self.cfg.dynamic.terminal_hidden_dim, 1)

        self.relu = torch.nn.ReLU()
        self.sm = torch.nn.Softmax(dim=1)
        self.sig = torch.nn.Sigmoid()

    def forward(self,state,action):
        """
        Note on orig. shapes: 
        - state is [-1, 8, 4, 4]
        - action looks like this 1, or [[1],[2],[3]..]
        We start by creating a m x 4 x 4 x 4, where for each m, 1 of the four channels (dim 1) is all 1s and then append this.
        """
        action = torch.tensor(action+1) / self.cfg.actions_size
        action = action.reshape(-1,1,1,1)
        action_plane = torch.zeros(state.shape[0],1, state.shape[2], state.shape[3]).to(self.device)
        action_plane += action
        
        # action_one_hot = TF.one_hot(torch.tensor(action).to(torch.int64),self.actions_size).reshape(-1,self.action_size, 1, 1).to(self.device)
        # print(torch.sum(action_one_hot))
        # action_plane += action_one_hot
        # action_plane = action_plane.to(self.device)
        
        x = torch.cat((state,action_plane),dim=1)
        ### so now we have a [m,12,4,4]
        x  = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        
        for block in self.resBlocks:
          x = block(x)
        state = x

        ##reward bit
        r = self.conv1x1_reward(x)
        r = self.bn3(r)
        r = self.relu(r)
        r = torch.flatten(r, start_dim=1)
        r = self.FC1(r)
        r = self.bn4(r)
        r = self.relu(r)
        r = self.FC2(r)
        r = self.sm(r)

        #terminal
        t = self.conv1x1_terminal(x)
        t = self.bn5(t)
        t = self.relu(t)
        t = torch.flatten(t, start_dim=1)
        t = self.FC1t(t)
        t = self.bn6(t)
        t = self.relu(t)
        t = self.FC2t(t)
        t = self.sig(t)
        return state, r, t 