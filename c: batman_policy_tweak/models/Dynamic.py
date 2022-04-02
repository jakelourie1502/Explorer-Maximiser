import sys


sys.path.append(".")
import torch
import torch.nn.functional as TF 
from global_settings import observable_size as env_size, actions_size, device, state_channels,first_layer_dyn_params, res_block_dyns,reward_conv_channels,reward_hidden_dim, reward_support
from models.res_block import resBlock
class Dynamic(torch.nn.Module):
    """
    Input: 
      state
    Notes:
     
    Outputs: 
      state

    """
    def __init__(self):
        self.action_size = actions_size
        self.first_layer_dyn_params = first_layer_dyn_params
        
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = state_channels + actions_size, out_channels = state_channels, 
                                                                    kernel_size= self.first_layer_dyn_params['kernel_size'],
                                                                    stride = self.first_layer_dyn_params['stride'],
                                                                    padding = self.first_layer_dyn_params['padding'])
        self.bn1 = torch.nn.BatchNorm2d(state_channels)
        self.conv2 = torch.nn.Conv2d(in_channels = state_channels, out_channels = state_channels, 
                                                                    kernel_size= 1,
                                                                    stride = 1,
                                                                    padding = 0)
        self.bn2 = torch.nn.BatchNorm2d(state_channels)
        self.resBlocks = torch.nn.ModuleList([resBlock(x) for x in res_block_dyns])
        #reward
        self.conv1x1_reward = torch.nn.Conv2d(in_channels=state_channels, out_channels=reward_conv_channels, kernel_size=1,padding=0, stride=1)
        self.bn3 = torch.nn.BatchNorm2d(state_channels)
        self.FC1 = torch.nn.Linear(env_size[0]*env_size[1]*state_channels, reward_hidden_dim)
        self.bn4 = torch.nn.BatchNorm1d(reward_hidden_dim)
        self.FC2 = torch.nn.Linear(reward_hidden_dim, reward_support[2])
        #terminal
        self.conv1x1_terminal = torch.nn.Conv2d(in_channels=state_channels, out_channels=reward_conv_channels, kernel_size=1,padding=0, stride=1)
        self.bn5 = torch.nn.BatchNorm2d(state_channels)
        self.FC1t = torch.nn.Linear(env_size[0]*env_size[1]*state_channels, reward_hidden_dim)
        self.bn6 = torch.nn.BatchNorm1d(reward_hidden_dim)
        self.FC2t = torch.nn.Linear(reward_hidden_dim, 1)

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

        action_plane = torch.zeros(state.shape[0],self.action_size, state.shape[2], state.shape[3]).to(device)
        action_one_hot = TF.one_hot(torch.tensor(action).to(torch.int64),actions_size).reshape(-1,self.action_size, 1, 1).to(device)
        action_plane += action_one_hot
        action_plane = action_plane.to(device)
        
        x = torch.cat((state,action_plane),dim=1)
        ### so now we have a [m,12,4,4]
        x  = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
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