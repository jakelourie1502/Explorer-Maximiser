from threading import Thread
import numpy as np
import torch
import torch.nn as nn
from training_and_data.replay_buffer import Replay_Buffer
from training_and_data.training import  loss_func_v, loss_func_p, loss_func_r, loss_func_proj, RDN_loss,loss_func_expV
from collections import deque
from utils import get_lr
import time


def compute_BL(algo): 
        algo.siam_log = deque([0], 200)
        pos_losses = {}
        neg_losses = {}
        for i in range(5):
            pos_losses[i] = deque([0],500)
            neg_losses[i] = deque([0],500)
        
        while algo.frame_count < algo.cfg.total_frames:

            if algo.training_flags.train_flag:
                algo.lr = get_lr(algo.training_step_counter,algo.cfg)
                algo.training_started = True
                algo.training_step_counter +=1
                for param_group in algo.optimizer.param_groups:
                    param_group['lr'] = algo.lr
                    
                algo.replay_buffer.purge() #keep replay buffer at reasonable size.
                
                ### getting target data
                sample_obs, indices, weights = algo.replay_buffer.get_sample(prioritised_sampling=algo.cfg.training.prioritised_replay)
                sample_obs = sample_obs.float()
                s = algo.ExpMaxTrain.representation(sample_obs)
                done_tensor_tmin1 = torch.zeros((len(indices),algo.K))
                done_tensor_same_t = torch.zeros((len(indices),algo.K))
                weights = torch.tensor(weights).to(algo.cfg.device_train).reshape(-1,1)
                loss = 0

                for k in range(algo.K):
                    action_index = np.array([algo.replay_buffer.action_log[x+k] for x in indices])
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

                    ## TERMINALITY
                    if algo.cfg.train_dones:
                        loss_dones_k = - (torch.mean((1-dones_k_tmin1_in_format)*(dones_k_same_t_in_format * torch.log(d+1e-4) + (1-dones_k_same_t_in_format)*torch.log(1-d+1e-4))))
                        loss+= algo.dones_coef * loss_dones_k

                    #### SIAM SECTION
                    if algo.cfg.use_siam:
                        o = torch.tensor(np.array([algo.replay_buffer.obs[x+k+1] for x in indices])).float()
                        w_grad_head = algo.ExpMaxTrain.project(s)
                        
                        with torch.no_grad():
                            reps = algo.ExpMaxTrain.representation(o)
                            stopped_proj = algo.ExpMaxTrain.project(reps, grad_branch = False)
                        loss_siam = loss_func_proj(stopped_proj, w_grad_head, dones_k_same_t_in_format,algo.cfg)
                        loss += algo.siam_coef * loss_siam
                        algo.siam_log.append(loss_siam.detach().cpu().numpy())
                    
                    #### POLICY
                    true_policy = torch.tensor(np.array([algo.replay_buffer.policy_logs[x+k] for x in indices])).to(algo.cfg.device_train).reshape(-1,algo.cfg.actions_size)
                    loss_Pk = loss_func_p(p, true_policy, dones_k_tmin1_in_format,weights,algo.cfg)
                    loss += loss_Pk
                    
                    #### VALUE AND REWARD (non-curiousity)
                    ## reanalyse values
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
                        
                        for idx, bv in zip(indices, boot_values[:, 0].cpu().numpy()):
                            algo.replay_buffer.n_step_returns_with_V[idx + k] = np.clip(algo.replay_buffer.n_step_returns[idx + k] + bv,algo.cfg.prediction.value_support[0],algo.cfg.prediction.value_support[1])
                        if np.random.uniform(0,10000) < 1:
                            print("Showing what the boot values, the n step returns and the n step returns with V look like")
                            print(boot_values[:10])
                            for idx in indices[:10]:
                                print(algo.replay_buffer.n_step_returns[idx+k], algo.replay_buffer.n_step_returns_with_V[idx + k])
                    else:
                        boot_values = 0
                        
                    if algo.cfg.value_only:
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

                # loss.backward()
                # algo.optimizer.step(); algo.optimizer.zero_grad()                        
                

                ### RDN SECTION
                if algo.training_step_counter % algo.cfg.training.main_to_rdn_ratio == 0 and algo.cfg.exploration_type == 'rdn':
                
                    if np.random.uniform(0,100) < 1: print("first move QE: ",algo.rdn_obj.new_ep_mu)
                    
                    sample_obs, indices, weights = algo.replay_buffer.get_sample(prioritised_sampling=False, exploration_sampling=algo.training_flags.expV_train_flag)
                    sample_obs = sample_obs.float()
                    s = algo.ExpMaxTrain.representation(sample_obs)
                    done_tensor_tmin1 = torch.zeros((len(indices),algo.K))
                    done_tensor_same_t = torch.zeros((len(indices),algo.K))
                    weights = torch.tensor(weights).to(algo.cfg.device_train).reshape(-1,1)
                    

                    ##### EXPV
                    expV_capturer = torch.zeros((algo.cfg.training.batch_size, algo.K, algo.cfg.prediction.expV_support[2]))
                    expR_capturer = torch.zeros((algo.cfg.training.batch_size, algo.K)).detach()
                    
                    for k in range(algo.K):
                        action_index = np.array([algo.replay_buffer.action_log[x+k] for x in indices])
                        rdn_vals = np.array([algo.replay_buffer.rdn_beta[x+k] for x in indices])
                        
                        if algo.cfg.detach_expV_calc:
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
                        rdn_random = algo.ExpMaxTrain.RDN(s).detach()
                        rdn_pred = algo.ExpMaxTrain.RDN_prediction(s.detach())
                        loss_rdn_k = algo.rdn_obj.train(rdn_random, rdn_pred, dones_k_tmin1_in_format, weights, k+1)
                        loss+= loss_rdn_k * algo.cfg.training.coef.rdn
                        expR_capturer[:,k] = torch.tensor(np.array([algo.replay_buffer.exp_r[x+k] for x in indices]))
                        
                    
                    ###expV finishing section
                    
                        
                    if algo.training_flags.expV_train_flag: #so this happens after 5,000 frames, before that we just give a small amount. we also then immediately start bootstrapping
                        true_expV_values = torch.zeros((algo.cfg.training.batch_size,algo.K))
                        true_expV_values += expR_capturer
                        for k in range(1,algo.K):
                            true_expV_values[:,:-k] += expR_capturer[:,k:] * (algo.cfg.exp_gamma**k)
                
                        obs_k_plus_N = torch.tensor(np.array([algo.replay_buffer.obs[x+algo.cfg.training.k] for x in indices])).float()          
                        K_plus_1_expV = algo.ExpMaxPlay.prediction(algo.ExpMaxPlay.representation(obs_k_plus_N),rdn_vals,only_predict_expV = True)[-1]
                        K_plus_1_expV = torch.tensor(K_plus_1_expV.detach().numpy() @ algo.support_expV) #bs, 1
                        K_plus_1_expV = (1-dones_k_same_t_in_format) * K_plus_1_expV 
                        if algo.cfg.use_new_episode_expV:
                            K_plus_1_expV += dones_k_same_t_in_format * algo.rdn_obj.new_ep_mu

                        K_plus_1_expV = torch.squeeze(K_plus_1_expV.detach())
                        for k in range(algo.K):
                            true_expV_values[:,-(k+1)] += K_plus_1_expV*algo.cfg.exp_gamma**(k+1) #here we are reversing our way through... the kth element will just be r+V* (one n step)
                        
                        true_expV_values = torch.clip(true_expV_values,*algo.cfg.prediction.expV_support[:2])

                    else:
                        true_expV_values = torch.randn((algo.cfg.training.batch_size, algo.K)) * 0.1
                        
                    for k in range(algo.K):
                        loss+= algo.cfg.training.coef.expV * loss_func_expV(expV_capturer[:,k],true_expV_values[:,k].reshape(-1,1),done_tensor_tmin1[:,k].reshape(-1,1),algo.cfg)
                        
                    algo.rdn_obj.update()
                
                if algo.training_step_counter % 2 == 0 and algo.cfg.exploration_type == 'episodic' and algo.thresholdhitter > 2:
                        
                    batch_size = algo.cfg.training.batch_size
                    
                    ep_loss_tracker = np.zeros((batch_size))
                    subject_obs, ep_ids_use, subject_index_points, true_index_points, fake_index_points,closeness_labels = algo.replay_buffer.sample_episodes(batch_size)
                    if np.random.uniform(0,30) < 1: 
                        print("unit testing: printing subject indexes, ture ones and takes ones to see it looks right: ", subject_index_points[:20], true_index_points[: 20], fake_index_points[: 20])
                    s = algo.ExpMaxTrain.representation(subject_obs)
                    for k in range(algo.K):
                        actions = np.array([algo.replay_buffer.episodes[x]['actions'][y+k] for x, y in zip(ep_ids_use, subject_index_points)])
                        s, _, d = algo.ExpMaxTrain.dynamic(s,actions)
                        s.register_hook(lambda grad: grad * 0.5)

                        # #DONES
                        # loss_dones_k = - (torch.mean(torch.log(1-d+1e-4)))
                        # loss+= algo.dones_coef * loss_dones_k

                        #SIAMS
                        # o = torch.tensor(np.array([algo.replay_buffer.episodes[x]['obs'][y+k+1] for x,y in zip(ep_ids_use, subject_index_points)])).float()
                        # w_grad_head = algo.ExpMaxTrain.project(s)
                        
                        # if algo.cfg.use_siam:
                            
                        #     with torch.no_grad():
                        #         reps = algo.ExpMaxTrain.representation(o.float())
                        #         stopped_proj = algo.ExpMaxTrain.project(reps, grad_branch = False)
                        #     loss_siam = loss_func_proj(stopped_proj, w_grad_head, torch.zeros((batch_size,1)),algo.cfg)
                        #     loss += algo.siam_coef * loss_siam

                        ### ENDS SIAM
                        
                        with torch.no_grad():
                            true_obs = torch.tensor(np.array([algo.replay_buffer.episodes[x]['obs'][y+k] for x, y in zip(ep_ids_use, true_index_points)])).float()
                            fake_obs = torch.tensor(np.array([algo.replay_buffer.episodes[x]['obs'][y+k] for x, y in zip(ep_ids_use, fake_index_points)])).float()
                            
                        subject_proj = algo.ExpMaxTrain.close_state_projection(s).repeat(2,1)
                        true_and_fake_states_together = torch.cat((true_obs,fake_obs),0)[:,-3:]
                        
                        
                        labels = torch.cat((torch.ones(batch_size),torch.zeros(batch_size)),0)
                        idx = torch.randperm(batch_size*2)
                        true_and_fake_states_together = true_and_fake_states_together[idx]
                        labels = labels[idx]
                        subject_proj_sorted = subject_proj[idx]
                        true_and_fake_projs = algo.ExpMaxTrain.close_state_projection_obs(true_and_fake_states_together)
                        true_fake_catted_with_proj = torch.cat((subject_proj_sorted, true_and_fake_projs),1)
                        
                        
                        predictions =  algo.ExpMaxTrain.close_state_classifer(true_fake_catted_with_proj).reshape(-1) #batch_size
                        
                        lossK = -torch.mean(\
                            labels*torch.log(predictions+1e-5) + (1-labels)*torch.log(1-predictions+1e-5))
                        # lossK = torch.mean((labels-predictions)**2)
                        loss+= lossK
                        pos_losses[k].append(torch.mean(predictions[labels==1]).detach().numpy())
                        neg_losses[k].append(torch.mean(predictions[labels==0]).detach().numpy())
                        
                        ep_loss_tracker+= lossK.detach().numpy() / 2
                        if k in [0,1]:
                            print("we're testing an obs printing thing on K: ",k)
                            print("Label: ", labels[0])
                            
                            stnum = algo.replay_buffer.episodes[ep_ids_use[idx[0] % batch_size]]['stnum'][subject_index_points[idx[0] % batch_size]+k+1]
                            if idx[0] < batch_size:
                                test_stnum = algo.replay_buffer.episodes[ep_ids_use[idx[0] % batch_size ]]['stnum'][true_index_points[idx[0] % batch_size]+k]
                            else:
                                test_stnum = algo.replay_buffer.episodes[ep_ids_use[idx[0] % batch_size ]]['stnum'][fake_index_points[idx[0] % batch_size]+k]
                            
                            print('prediction: ', predictions[0].detach().numpy())
                            print("true state number: ", stnum)
                            print("comparator state stnum: ", test_stnum)
                            
                        if k == 0:
                                    with torch.no_grad():
                                        algo.ExpMaxTrain.eval()
                                        s1 = algo.ExpMaxTrain.representation(subject_obs[idx[0]%batch_size].unsqueeze(0).float())
                                        s1, _, _ = algo.ExpMaxTrain.dynamic(s1,actions[idx[0]%batch_size])

                                        TFobs = true_and_fake_states_together[0].unsqueeze(0).float()
                                        subject_proj = algo.ExpMaxTrain.close_state_projection(s1)
                                        true_and_fake_proj = algo.ExpMaxTrain.close_state_projection_obs(TFobs[:,-3:])
                                        print("label: ", labels[0])
                                        predictionss = algo.ExpMaxTrain.close_state_classifer(torch.cat((subject_proj, true_and_fake_proj),1))
                                        print("prediction: ", predictionss)
                                        state_for_checking = s[idx[0]%batch_size].unsqueeze(0)
                                        proj_for_checking = subject_proj_sorted[idx[0]].unsqueeze(0)
                                        tf_proj_for_checking = true_and_fake_projs[idx[0]%batch_size].unsqueeze(0)
                                        
                                        print("state diffs: ", torch.sum(torch.abs(state_for_checking - s1)))
                                        print("cosine sim: ", torch.nn.CosineSimilarity(dim=1)(s1.reshape(1,-1),state_for_checking.reshape(1,-1)))
                                        print('proj diffs: ', torch.sum(torch.abs(subject_proj.reshape(-1) - proj_for_checking.reshape(-1))))
                                        print('cosine projs: ', torch.nn.CosineSimilarity(dim=1)(subject_proj.reshape(1,-1), proj_for_checking.reshape(1,-1)))
                                        print('tf proj diffs: ', torch.sum(torch.abs(true_and_fake_proj.reshape(-1) - tf_proj_for_checking.reshape(-1))))
                                        print('tf cosine projs: ', torch.nn.CosineSimilarity(dim=1)(true_and_fake_proj.reshape(1,-1), tf_proj_for_checking.reshape(1,-1)))
                                        
                                        # predictions_positive_pairs =  algo.ExpMaxTrain.close_state_classifer(torch.cat((subject_proj,true_proj),1)).detach().numpy()
                                        # predictions_negative_pairs =  algo.ExpMaxTrain.close_state_classifer(torch.cat((subject_proj,fake_proj),1)).detach().numpy()
                                        # print("Checker on if a 1 batch with neg and positive pair: ", predictions_positive_pairs, predictions_negative_pairs)
                                        # print("state number, true state number and fake state number: ", con(stnum), con(t_stnum), con(f_stnum))
                                    algo.ExpMaxTrain.train()
                    if np.random.uniform(0,25) < 1:
                        print("pos losses: ", {k: np.mean(pos_losses[k]) for k in range(5)})
                        print("neg losses: ", {k: np.mean(neg_losses[k]) for k in range(5)})
                        print('sample 1s: ', predictions[labels==1][:10].detach().numpy())
                        print('neg positives: ', predictions[labels==0][:10].detach().numpy())
                        
                algo.scores.update()
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