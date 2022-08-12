import gym
import skvideo.io
from PIL import Image
import time
import matplotlib.pyplot as plt
import cv2
from collections import deque
import sys
sys.path.append('..')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from math import factorial as F
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
from pil_video import make_video

from game_play.wrappers import make_atari
from gym.wrappers import video_recorder
from preMadeEnvironments.Lakes6x6 import easy_version6x6, medium_version1_6x6, medium_version2_6x6, hard_version6x6
from preMadeEnvironments.Lakes8x8 import easy_version8x8, medium_version1_8x8, medium_version2_8x8, hard_version8x8
from preMadeEnvironments.Lake_Erroneous import erroneous, erroneous_with_second_goal
from preMadeEnvironments.RaceWorld import medium
from preMadeEnvironments.Key_Envs import key1
# from game_play.frozen_lakeGym_Image import gridWorld as Env
from utils import global_expl, Scores
# from game_play.frozen_lake_KEY import gridWorld as Env
from game_play.Car_Driving_Env import RaceWorld as Env
# from game_play.frozen_lakeGym_Image import gridWorld as Env
from game_play.play_episode import Episode
from game_play.mcts import MCTS, Node
from training_and_data.training import RDN_loss
from config import child
# sys.stdout = open("playing_game.txt","w")
class Ep_counter:
    def __init__(self):
        self.ids = []
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', type=int, default=0)
parser.add_argument('--algo', type=int, default=0)
args = parser.parse_args()
from config import Config
cfg = Config(args.env, args.algo)

from utils import vector2d_to_tensor_for_model
from game_play.play_episode import Episode
siam_scores = np.zeros((cfg.training.k)) + 0.5

class Ep_counter:
    def __init__(self):
        self.ids = []

ep_c = Ep_counter()
ExpMax = torch.load('../saved_models/jake_zeroFA_CARMED_1',map_location=torch.device('cpu'))
rdn_obj = RDN_loss(cfg)
rdn_obj.deki_stats = ExpMax.rdn_dekis
rdn_obj.new_ep_mu = 0.

move_count = 0
q_tracker = child()
q_tracker.end_states = 0
q_tracker.non_end_states = 0

for i in range(100):
    ExpMax.eval()
    with torch.no_grad():
        ep = Episode(ExpMax, cfg, Scores(cfg), Ep_counter(),1e6, rdn_obj = rdn_obj, test_mode=False, q_tracker=q_tracker)
        ep.rdn_beta = 0.0
        ep.actor_id = 0.
        ep.actor_policy = 0.
        ep.curr_best_score = 0.
        mcts  = MCTS(ep, epoch = 1e6, pick_best=True,view=False)
        if cfg.atari_env:
            env = gym.make(cfg.env)
        else:
            env = Env(cfg)
        # vid = video_recorder.VideoRecorder(env, path = 'vid.mp4',enabled=True)
        # print(vid.__dict__)
        # print(vid.env.metadata.get['video.frames_per_second'])
        obs_deque = deque([], cfg.deque_length)
        done = False
        obs= env.reset()
        # out = cv2.VideoWriter('goodVid.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 2, (96,96))
        list_of_frames = []
        if cfg.atari_env:
            obs = cv2.resize(obs, (cfg.image_size[0],cfg.image_size[1]), interpolation=cv2.INTER_AREA)
            obs = np.transpose((obs.astype(np.float32) / 255),(2,0,1))
        else:
            obs = np.transpose(obs,(1,2,0))
            im = Image.fromarray(np.uint8(obs))
            
            list_of_frames.append(obs)
            # out.write(obs)
            obs = cv2.resize(obs.astype(np.float32) / 255, (48,48), interpolation=cv2.INTER_AREA)
            obs = np.transpose(obs.astype(np.float32),(2,0,1))
        for _ in range(cfg.deque_length // 2):
            obs_deque.append(np.zeros_like(obs))
            obs_deque.append(np.expand_dims(np.zeros_like(obs[0]),0))
        obs_deque.append(obs) #so this is now a list of 3, a (3, h ,w), a (1, h ,w) and a (3, h, w)
        print(len(obs_deque))
        Q = 1
        Qe = 0.5
        ep.running_reward=0
        
        
        
        while not done:
            # vid.capture_frame()
            
            ep.move_number += 1
            move_count +=1
            o = vector2d_to_tensor_for_model(np.concatenate(obs_deque,0)) #need to double unsqueeze here to create fictional batch and channel dims
            state = ExpMax.representation(o.float())
            ep.state_vectors.append((ExpMax.close_state_projection(state)))
            root_node = Node('null', Q = Q, Qe = Qe)
            root_node.state = state
            root_node.SVs = deque([], 10)
            policy, action, Q,  v, Qe, imm_nov, expected_reward = mcts.one_turn(root_node) ## V AND rn_v is used for bootstrapping
            act = action
            Qe = max(Qe,0) ## at the beginning, if expR is often negative, this compounds expV. so the first node selected will be much less negative after 1 sim.
            obs, reward, done, stnum  = env.step(action)
            
            if cfg.atari_env:
                obs = cv2.resize(obs, (cfg.image_size[0],cfg.image_size[1]), interpolation=cv2.INTER_AREA)
                obs = np.transpose((obs.astype(np.float32) / 255),(2,0,1))
            else:
                obs = np.transpose(obs,(1,2,0))
                # plt.close()
                # plt.imshow(obs)
                # plt.show()
                # print(obs.shape)
                # out.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
                list_of_frames.append(obs)
                # out.write(obs)
                obs = cv2.resize(obs.astype(np.float32) / 255, (48,48), interpolation=cv2.INTER_AREA)
                obs = np.transpose(obs.astype(np.float32),(2,0,1))
            if cfg.store_prev_actions:
                action_frame = np.expand_dims(np.zeros_like(obs[0]),0) + (act+1) / cfg.actions_size 
                obs_deque.append(action_frame)
            obs_deque.append(obs)
        # out.release()
        # a = make_video(list_of_frames, 2 , play_video=False)
        # print(a)
        # cv2.destroyAllWindows()
        # vid.close()    
        #     # print(obs_deque)
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video = cv2.VideoWriter('goodVid.mp4',fourcc, 2, (96,96))

        outputdata = np.stack(list_of_frames,0).astype(np.uint8)
        print(outputdata.shape)
        fps = 2
        # create writer using FFmpegWriter
        try:
            os.mkdir('videos')
        except: pass
        writer = skvideo.io.FFmpegWriter(f'videos/outputvideo{i}.mp4', 
                        inputdict={'-r': str(fps)},
                        outputdict={'-r': str(fps), '-c:v': 'libx264', '-preset': 'ultrafast', '-pix_fmt': 'yuv444p'})
        
        # skvideo.io.vwrite("outputvideo.mp4", outputdata, )
        for image in list_of_frames:
            writer.writeFrame(image)
        writer.close()

            # plt.imshow(image)
            # plt.close()
            # video.write(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))

cv2.destroyAllWindows()
# video.release()