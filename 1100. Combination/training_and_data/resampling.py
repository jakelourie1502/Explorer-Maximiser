
from logging import root
import numpy as np
import torch
import time
from game_play.play_episode import Episode
from game_play.mcts import MCTS, Node
from threading import Thread
def resample_trajectories(algo):
    while algo.frame_count < algo.cfg.total_frames:
        if algo.training_flags.resampling_flag:
            print("resampling")
            algo.resampling_step_counter += 1
            resampling_size = algo.cfg.training.batch_size
            obs, indices, weights = algo.replay_buffer.get_sample(prioritised_sampling=True, batch_size = resampling_size, resampling=True)
            predicted_v = [algo.replay_buffer.predicted_v[x] for x in indices]
            with torch.no_grad():
                state = algo.ExpMaxTrain.representation(obs.float()).to(algo.cfg.device_selfplay)
                policies_true = np.zeros((resampling_size, algo.cfg.actions_size))
                Q_vals = torch.zeros((resampling_size, 1))
                best_actions = torch.zeros((resampling_size))
                rdn_vals = torch.zeros((resampling_size))
            
            def re_trace(index, init_pass=True):
                with torch.no_grad():
                    s = state[index]
                    episode = Episode(algo.ExpMaxPlay,algo.cfg,scores= algo.scores, ep_counter = algo.ep_counter, epoch = algo.frame_count, rdn_obj = algo.rdn_obj, test_mode=False, q_tracker=algo.q_tracker, current_best_score=algo.curr_best_score, resampling=True)
                    episode.actor_policy = 0
                    if not init_pass:
                        episode.rdn_beta = rdn_vals[index]
                    mcts = MCTS(episode = episode, epoch = algo.frame_count, pick_best=episode.pick_best)
                    root_node = Node('null',Q=predicted_v[index], Qe=0.)
                    root_node.state = s.unsqueeze(0)
                    policy, action, Q,  v, Qe, imm_nov = mcts.one_turn(root_node)
                    if algo.cfg.training.resampling_use_max:
                        if np.random.uniform(0,5) < 1:
                            print("OLD Q: ", Q)
                            Q = mcts.mm.maximum
                            print("NEW Q: ", Q)
                        else: Q = mcts.mm.maximum
                    best_action = np.argmax([x.Q for x in root_node.children])
                    best_actions[index] = best_action
                    rdn_vals[index] = episode.rdn_beta
                    Q_val = np.clip(Q, algo.cfg.prediction.value_support[0], algo.cfg.prediction.value_support[1])
                    Q_vals[index] = Q_val
                    if init_pass: 
                        policies_true[index] = policy

            for i in range(resampling_size):
                re_trace(i, init_pass=True)
            
            if algo.cfg.training.resampling_assess_best_child:
                with torch.no_grad():
                    
                    state, r, d = algo.ExpMaxTrain.dynamic(state, best_actions)
                    r = torch.tensor(r.detach().numpy() @ algo.support_full_rewards) #128 x 1, r_support
                    for i in range(resampling_size):
                        re_trace(i, init_pass=False)
                    Q_vals = torch.clip(Q_vals*(1-d)*algo.cfg.gamma + r, algo.cfg.prediction.value_support[0], algo.cfg.prediction.value_support[1])

            policies_true = torch.tensor(policies_true)
            Q_vals = Q_vals.reshape(-1,1)
            algo.RS.obs.append(obs)
            algo.RS.weights.append(torch.tensor(weights).reshape(-1,1))
            algo.RS.Qs.append(Q_vals)
            algo.RS.pols.append(policies_true)
            algo.RS.actions.append(best_actions)
            algo.RS.rdns.append(rdn_vals)
        else:
            time.sleep(60)
            print("pausing resampling")