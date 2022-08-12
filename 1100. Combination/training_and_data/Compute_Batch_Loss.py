from threading import Thread
import numpy as np
import torch
from training_and_data.training import  loss_func_v, loss_func_p, loss_func_r, loss_func_proj,loss_func_expV
from collections import deque
from utils import get_lr
import time


def compute_BL(algo): 
        """
        End to end training function for all versions of the algorithm"""

        algo.siam_log = deque([0], 200)
        while algo.frame_count < algo.cfg.total_frames:
            if algo.training_flags.train_flag: #this flag ensures the right ratio of frames-trained-to-played
                algo.lr = get_lr(algo.training_step_counter,algo.cfg) #lr ramps up slowly as per efficient zero.
                algo.training_started = True
                algo.training_step_counter +=1
                for param_group in algo.optimizer.param_groups:
                    param_group['lr'] = algo.lr
                    
                algo.replay_buffer.purge() #keep replay buffer at reasonable size.
                
                ### getting target data
                sample_obs, indices, weights = algo.replay_buffer.get_sample(prioritised_sampling=algo.cfg.training.prioritised_replay)
                sample_obs = sample_obs.float()
                s = algo.ExpMaxTrain.representation(sample_obs)
                
                ### Get an empty array for dones, which are used as a mask.
                done_tensor_tmin1 = torch.zeros((len(indices),algo.K))
                done_tensor_same_t = torch.zeros((len(indices),algo.K))
                weights = torch.tensor(weights).to(algo.cfg.device_train).reshape(-1,1) #these are the prioritisation sampling weights to avoid sampling bias.
                loss = 0

                for k in range(algo.K):
                    ###### CAPTURING DONE MASKS (need two sets, as some variables require the prior k done, and some require this k done)
                    ### SAME period
                    dones_k_same_t = np.array([algo.replay_buffer.done_logs[x+k] for x in indices])
                    dones_k_same_t = torch.maximum(torch.tensor(dones_k_same_t), done_tensor_same_t[:, k-1]).to(algo.cfg.device_train)
                    done_tensor_same_t[:, k] = dones_k_same_t
                    dones_k_same_t_in_format = dones_k_same_t.reshape(-1,1)
                    
                    ### PRIOR period
                    if k == 0:
                        dones_k_tmin1 = torch.zeros((algo.cfg.training.batch_size))
                    else:
                        dones_k_tmin1 = np.array([algo.replay_buffer.done_logs[x+k-1] for x in indices])
                        dones_k_tmin1 = torch.maximum(torch.tensor(dones_k_tmin1), done_tensor_tmin1[:, k-1]).to(algo.cfg.device_train)
                        done_tensor_tmin1[:, k] = dones_k_tmin1
                    dones_k_tmin1_in_format = dones_k_tmin1.reshape(-1,1)


                    action_index = np.array([algo.replay_buffer.action_log[x+k] for x in indices])
                    
                    ### Action prediction section for episodic novelty.
                    # Predict next action based on next observation and current state
                    if algo.cfg.exploration_type in ['episodic','both'] and algo.training_step_counter % 2 == 0:
                        o = torch.tensor(np.array([algo.replay_buffer.obs[x+k+1] for x in indices])).float()
                        obs_next = o[:, -3:] #unstacked observation

                        #get the controllable states
                        projected_state = algo.ExpMaxTrain.close_state_projection(s) #creates e(s) - > s^c
                        projected_obs = algo.ExpMaxTrain.close_state_projection_obs(obs_next) # create e^o(o) -> o^c
                        
                        #predict the action
                        predicted_action = algo.ExpMaxTrain.close_state_classifer(torch.cat((projected_state, projected_obs),1))
                        
                        next_actions = torch.tensor(action_index.reshape(-1,1)) #true action
                        next_actions_hot = torch.nn.functional.one_hot(next_actions, algo.cfg.actions_size).squeeze(1)
                        loss_action_pred = -torch.mean((1-dones_k_same_t_in_format)*(next_actions_hot * torch.log(predicted_action +1e-5)))
                        loss += loss_action_pred

                        if algo.cfg.contrast_vector and algo.training_step_counter % 2 == 0 and k!=0:
                            # Contrast state vector with previous period state vector and random state vector
                            shuffled_idx = torch.randperm(algo.cfg.training.batch_size) #create random state vectors for negative contrast
                            shuffled_obs_projs = last_state_proj[shuffled_idx]
                            pos_sims = algo.ExpMaxTrain.contrast_two_state_vecs(projected_state, last_state_proj) #returns simTRUE / 0.07
                            neg_sims = algo.ExpMaxTrain.contrast_two_state_vecs(projected_state, shuffled_obs_projs) #returns sim1FALSE / 0.07
                            
                            ## do the contrastive loss as per original paper on CL
                            catted_sims = torch.exp(torch.cat((pos_sims,neg_sims),1))  
                            catted_sims = catted_sims / torch.sum(catted_sims,1,keepdim=True) #SOFTMAX
                            loss_sim_vecs = -torch.mean(torch.log(catted_sims[:,0]+1e-4)) # we only add the loss from the positive branch as per paper.
                            loss+= loss_sim_vecs

                        last_state_proj = projected_state #allows us to use the last state as contrast
                        
                        ### UNIT TESTING to see how good the algorithm is at predicting the action taken.
                        if np.random.uniform(0,200) < 1:
                            print("actions average: ") 

                            for i in range(algo.cfg.actions_size):
                                print('K: ', k, ' action ', i, ': ', torch.mean((1-dones_k_same_t_in_format)*next_actions_hot[:,i].unsqueeze(1)*predicted_action,0)\
                                    /torch.mean((1-dones_k_same_t_in_format)*(next_actions_hot[:,i]).unsqueeze(1).float()))

                    ### Recurrent inference section (extract value estimates from model)
                    rdn_vals = np.array([algo.replay_buffer.rdn_beta[x+k] for x in indices])
                    actor_mask = torch.tensor(np.array([algo.replay_buffer.actor_id[x+k] for x in indices]).reshape(-1,1))
                    predicted_values = algo.ExpMaxTrain.prediction(s,rdn_vals)
                    if algo.cfg.use_two_heads:
                        maxi_p, expl_p, maxi_v,  expl_v , expV = predicted_values
                        p = actor_mask * expl_p + (1-actor_mask) * maxi_p
                        v = actor_mask * expl_v + (1-actor_mask) * maxi_v
                    else:
                        p, v, expV = predicted_values

                    s, r, d = algo.ExpMaxTrain.dynamic(s,action_index)
                    s.register_hook(lambda grad: grad * 0.5)
                    #note: episodic memory function can be done with dones last period because if it's dones this period, we still have an action to predict.

                    ## TERMINALITY <- predicting whether episode has finished
                    if algo.cfg.train_dones:
                        loss_dones_k = - (torch.mean((1-dones_k_tmin1_in_format)*\
                            (dones_k_same_t_in_format * torch.log(d+1e-4) +\
                            (1-dones_k_same_t_in_format)*torch.log(1-d+1e-4))))
                        loss+= algo.dones_coef * loss_dones_k

                    #### SimSiam SECTION
                    if algo.cfg.use_siam:
                        o = torch.tensor(np.array([algo.replay_buffer.obs[x+k+1] for x in indices])).float()
                        w_grad_head = algo.ExpMaxTrain.project(s)
                        
                        with torch.no_grad():
                            reps = algo.ExpMaxTrain.representation(o)
                            stopped_proj = algo.ExpMaxTrain.project(reps, grad_branch = False)
                        loss_siam = loss_func_proj(stopped_proj, w_grad_head, dones_k_same_t_in_format,algo.cfg)
                        loss += algo.siam_coef * loss_siam
                        algo.siam_log.append(loss_siam.detach().cpu().numpy())
                    
                    #### Learn policy
                    true_policy = torch.tensor(np.array([algo.replay_buffer.policy_logs[x+k] for x in indices])).to(algo.cfg.device_train).reshape(-1,algo.cfg.actions_size)
                    loss_Pk = loss_func_p(p, true_policy, dones_k_tmin1_in_format,weights,algo.cfg)
                    loss += loss_Pk
                    
                    #### Extrinsic value and reward section
                    # reanalyse values to create updated estimate of bootstrapped N-STEP returns
                    if algo.frame_count >= algo.cfg.calc_n_step_rewards_after_frames:
                        with torch.no_grad():
                            obs_k_plus_N = torch.tensor(np.array([algo.replay_buffer.obs[x+k+algo.cfg.N_steps_reward] for x in indices])).float()          
                            predicted_values = algo.ExpMaxPlay.prediction(algo.ExpMaxPlay.representation(obs_k_plus_N),rdn_vals)
                            if algo.cfg.use_two_heads:
                                _, _,v_reanalyse_maxi, v_reanalyse_expl,_ = predicted_values
                                v_reanalyse_maxi = torch.tensor(v_reanalyse_maxi.detach().numpy() @ algo.support_full_values)
                                v_reanalyse_expl = torch.tensor(v_reanalyse_expl.detach().numpy() @ algo.support_full_values)
                                v_reanalyse = torch.maximum(v_reanalyse_expl, v_reanalyse_maxi)
                            else:
                                v_reanalyse = torch.tensor(predicted_values[1].detach().numpy() @ algo.support_full_values)
                            k_ep_id = np.array([algo.replay_buffer.ep_id[x+k] for x in indices])
                            k_plus_N_ep_id = np.array([algo.replay_buffer.ep_id[x+k+algo.cfg.N_steps_reward] for x in indices])
                            mask = (k_ep_id == k_plus_N_ep_id).reshape(-1,1)
                            boot_values = torch.tensor(v_reanalyse * mask) * (algo.cfg.gamma ** algo.cfg.N_steps_reward)
                        
                        for idx, bv in zip(indices, boot_values[:, 0].cpu().numpy()): #update replay buffer
                            algo.replay_buffer.n_step_returns_with_V[idx + k] = np.clip(algo.replay_buffer.n_step_returns[idx + k] + bv,algo.cfg.prediction.value_support[0],algo.cfg.prediction.value_support[1])
                        if np.random.uniform(0,10000) < 1:
                            print("Showing what the boot values, the n step returns and the n step returns with V look like")
                            print(boot_values[:10])
                            for idx in indices[:10]:
                                print(algo.replay_buffer.n_step_returns[idx+k], algo.replay_buffer.n_step_returns_with_V[idx + k])
                    else:
                        boot_values = 0
                        
                    if algo.cfg.value_only: #obsolete <- model currently works just for reward and value.
                        true_values = torch.tensor(np.array([algo.replay_buffer.n_step_returns[x] for x in indices])).to(algo.cfg.device_train).reshape(-1,1) #here, when we just use value, we don't use the dones in the value calculation.
                    else:
                        true_values = torch.tensor(np.array([algo.replay_buffer.n_step_returns_with_V[x+k] for x in indices])).to(algo.cfg.device_train).reshape(-1,1) 
                        true_rewards = torch.tensor(np.array([algo.replay_buffer.reward_logs[x+k] for x in indices])).to(algo.cfg.device_train).reshape(-1,1)
                        loss_Rk = loss_func_r(r, true_rewards, dones_k_tmin1_in_format, weights,algo.cfg)
                        loss += loss_Rk
                    
                    loss_Vk = loss_func_v(v, true_values, dones_k_tmin1_in_format,weights,algo.cfg)
                    loss += loss_Vk * algo.value_coef

                if np.random.uniform(0,50) < 1 or algo.training_step_counter == 2:
                    print("siam score: ", np.mean(algo.siam_log))

                ### RDN SECTION
                if algo.training_step_counter % algo.cfg.training.main_to_rdn_ratio == 0 and algo.cfg.exploration_type in ['rdn','vNov_ablation','both']:
                
                    if np.random.uniform(0,100) < 1: print("first move QE: ",algo.rdn_obj.new_ep_mu)
                    
                    ### we don't use prioritised sampling for rnd
                    sample_obs, indices, weights = algo.replay_buffer.get_sample(prioritised_sampling=False, exploration_sampling=algo.training_flags.expV_train_flag)
                    sample_obs = sample_obs.float()
                    s = algo.ExpMaxTrain.representation(sample_obs)
                    done_tensor_tmin1 = torch.zeros((len(indices),algo.K))
                    done_tensor_same_t = torch.zeros((len(indices),algo.K))
                    weights = torch.tensor(weights).to(algo.cfg.device_train).reshape(-1,1)
                     
                    ##### EXPV
                    # in case we want to use on policy rnd (even though in the final best version we don't), we collect true r^{nov} for expV each k
                    expV_capturer = torch.zeros((algo.cfg.training.batch_size, algo.K, algo.cfg.prediction.expV_support[2]))
                    expR_capturer = torch.zeros((algo.cfg.training.batch_size, algo.K)).detach()
                    
                    for k in range(algo.K):
                        action_index = np.array([algo.replay_buffer.action_log[x+k] for x in indices])
                        rdn_vals = np.array([algo.replay_buffer.rdn_beta[x+k] for x in indices])
                        
                        if algo.cfg.detach_expV_calc:
                            #so predict expV for that K
                            expV = algo.ExpMaxTrain.prediction(s.detach(),rdn_vals, only_predict_expV = True)[-1]
                        else:
                            expV = algo.ExpMaxTrain.prediction(s,rdn_vals,only_predict_expV = True)[-1]
                        
                        expV_capturer[:,k] = expV
                        s, r, _ = algo.ExpMaxTrain.dynamic(s,action_index)
                        s.register_hook(lambda grad: grad * 0.5)
                        
                        ###### CAPTURING DONE MASKS (need two sets, as some variables require the prior k done, and some require this k done)
                        ## DONES
                        #same period
                        dones_k_same_t = np.array([algo.replay_buffer.done_logs[x+k] for x in indices])
                        dones_k_same_t = torch.maximum(torch.tensor(dones_k_same_t), done_tensor_same_t[:, k-1]).to(algo.cfg.device_train)
                        done_tensor_same_t[:, k] = dones_k_same_t
                        dones_k_same_t_in_format = dones_k_same_t.reshape(-1,1)

                        #last period
                        if k == 0:
                            dones_k_tmin1 = done_tensor_tmin1[:, 0].to(algo.cfg.device_train)
                        else:
                            dones_k_tmin1 = np.array([algo.replay_buffer.done_logs[x+k-1] for x in indices])
                            dones_k_tmin1 = torch.maximum(torch.tensor(dones_k_tmin1), done_tensor_tmin1[:, k-1]).to(algo.cfg.device_train)
                            done_tensor_tmin1[:, k] = dones_k_tmin1
                        dones_k_tmin1_in_format = dones_k_tmin1.reshape(-1,1)
                        
                        ##### EXPR
                        #teach the RND network
                        rdn_random = algo.ExpMaxTrain.RDN(s).detach()
                        rdn_pred = algo.ExpMaxTrain.RDN_prediction(s.detach())
                        loss_rdn_k = algo.rdn_obj.train(rdn_random, rdn_pred, dones_k_tmin1_in_format, weights, k+1)
                        loss+= loss_rdn_k * algo.cfg.training.coef.rdn
                        expR_capturer[:,k] = torch.tensor(np.array([algo.replay_buffer.exp_r[x+k] for x in indices])) #put true value of rnd from buffer into the expR capturer
                        
                    
                    ###Actually training v^{nov}
                    if algo.training_flags.expV_train_flag and algo.cfg.exploration_type != 'vNov_ablation': #so this happens after 5,000 frames, before that we just give a small amount. we also then immediately start bootstrapping
                        ## create an empty array of batch size x K
                        true_expV_values = torch.zeros((algo.cfg.training.batch_size,algo.K))
                        true_expV_values += expR_capturer #add r^{nov} values each k in K steps.
                        for k in range(1,algo.K):
                            true_expV_values[:,:-k] += expR_capturer[:,k:] * (algo.cfg.exp_gamma**k) #this calculates the N step.
                
                        ## to get v^{nov}_{t+K} for bootstrapping
                        obs_t_plus_K = torch.tensor(np.array([algo.replay_buffer.obs[x+algo.cfg.training.k] for x in indices])).float()          
                        t_plus_K_expV = algo.ExpMaxPlay.prediction(algo.ExpMaxPlay.representation(obs_t_plus_K),rdn_vals,only_predict_expV = True)[-1]
                        t_plus_K_expV = torch.tensor(t_plus_K_expV.detach().numpy() @ algo.support_expV) #bs, 1
                        t_plus_K_expV = (1-dones_k_same_t_in_format) * t_plus_K_expV
                        if algo.cfg.use_new_episode_expV:
                            t_plus_K_expV += dones_k_same_t_in_format * algo.rdn_obj.new_ep_mu

                        t_plus_K_expV = torch.squeeze(t_plus_K_expV.detach())
                        for k in range(algo.K):
                            #add the tPlusKv^nov to each of the N stepped values.
                            true_expV_values[:,-(k+1)] += t_plus_K_expV*algo.cfg.exp_gamma**(k+1) #here we are reversing our way through... the kth element will just be r+V* (one n step)
                        
                        true_expV_values = torch.clip(true_expV_values,*algo.cfg.prediction.expV_support[:2])

                    else:
                        true_expV_values = torch.randn((algo.cfg.training.batch_size, algo.K)) * 0.1
                        
                    for k in range(algo.K):
                        ### Train the loss for each of the k steps
                        loss+= algo.cfg.training.coef.expV * loss_func_expV(expV_capturer[:,k],true_expV_values[:,k].reshape(-1,1),done_tensor_tmin1[:,k].reshape(-1,1),algo.cfg)
                        
                    algo.rdn_obj.update() #update the averages used in the running totals for rnd values.
                
                algo.scores.update() #update averages for explorer and maximiser extrinsic reward performance
            #### END OF LOSS ACCUMULATION
                loss.backward()
                algo.optimizer.step(); algo.optimizer.zero_grad()
                
                if algo.training_step_counter % algo.cfg.update_play_model == 0:
                    algo.save_and_load_model()
                

            else:
                
                print("main train batch thing paused")
                if algo.training_flags.add_more_self_play_workers and algo.thread_count < algo.cfg.training.max_workers:
                    print("add a thread")
                    actor_thread = Thread(target=algo.actor_wrapper, args=())
                    actor_thread.start()
                    algo.thread_count+=1
                    print(f"Adding thread: now have {algo.thread_count} threads")
                time.sleep(1*20)