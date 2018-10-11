import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)   # replay buffer size
MIN_BUF_SIZE = 4*64      # minimum buffer size to start learning
BATCH_SIZE = 64          # minibatch size
GAMMA = 0.99             # discount factor
TAU = 1e-3               # for soft update of target parameters
LR = 5e-4                # learning rate 
UPDATE_EVERY = int(1000) # Num of steps between two censecutive learning phases
NUM_LEARNS = int(1000)   # Num of learnings conducted in a learning phase
SORT_EVERY = 100         # Num of learnings between memory sort based on deltas
BETA_ANNEALING = 0.001   # Update value for the linear annealing for beta
ALPHA_ANNEALING = 0.001  # Update value for the linear annealing for alpha

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

PARAMETER_ANNEALING=False

class AgentPriority():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, hidden_layers, lr=5e-4, alpha=0.5, beta=0.4):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            hidden_layers (list[int, int, ...]): size of hidden layers
            lr (float): learning rate
            alpha (float (0<=alpha<=1)): parameter alpha for priority
            beta (float (0<=beta<=1)): parameter for importance sampling weight
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        

        # Q-Network
        self.lr = lr
        self.qnetwork_local = QNetwork(state_size, action_size, self.seed, hidden_layers).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, self.seed, hidden_layers).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.alpha = alpha
        self.beta = beta
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, self.alpha, self.beta)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # discount
        self.gamma = GAMMA
        
        self.checkpoint = {"input_size": self.state_size,
                           "output_size": self.action_size,
                           "hidden_layers": [each.out_features for each in self.qnetwork_local.hidden_layers],
                           "state_dict": self.qnetwork_local.state_dict()}
        self.checkpointfile = 'priority_ddqn.pth'
        
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory       
        delta = self.comp_delta(state, action, reward, next_state, done)
        self.memory.add(state, action, reward, next_state, done, delta)
        
        # Learn NUM_LEARNS times par every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) >= MIN_BUF_SIZE:
            self.memory.set_priority_params(self.alpha, self.beta)
            for i in range(NUM_LEARNS):
                if i%SORT_EVERY == 0:
                    # Sort memory based on delta every SORT_EVERY learnings
                    self.memory.argsort_deltas()
                    
                    # Update q_target with q_local
                    self.update_qtarget()
                    
                    # If PARAMETER_ANNEALING is set to True,anneal alpha & beta.
                    if PARAMETER_ANNEALING:
                        self.parameter_anneal()
                    
                experiences, weights, mem_idxs = self.memory.sample()
                self.learn(experiences, weights, mem_idxs)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).astype(np.int32)
        else:
            return random.choice(np.arange(self.action_size)).astype(np.int32)

    def learn(self, experiences, weights, mem_idxs):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            mem_idxs (list of ints): indices in the replay buffer corresponding to
                                     the given experiences (used to update delta)
        """
        states, actions, rewards, next_states, dones = experiences

        # Get argmax of Q values (for next states) from Q_local model
        Q_local_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        
        # Evaluate that actions with Q_target model
        Q_targets_next = self.qnetwork_target(next_states).gather(1, Q_local_actions).detach()
        
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # update deltas in self.memory
        deltas = (Q_targets - Q_expected).detach().cpu().numpy()
        self.memory.update_deltas(deltas, mem_idxs)
        
        # Compute loss
        loss = F.mse_loss(weights*Q_expected, weights*Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def update_qtarget(self):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(local_param.data)

    def comp_delta(self, state, action, reward, next_state, done):
        """Compute delta given an experience
        delta = reward + gamma*argmax_action(Q_target(next_state, a)) - Q_local(state, action)
        """
        state_ts = torch.from_numpy(np.expand_dims(state,0)).float().to(device)
        action_ts = torch.from_numpy(np.array([[action]])).long().to(device)
        reward_ts = torch.from_numpy(np.array([[reward]])).float().to(device)
        next_state_ts = torch.from_numpy(np.expand_dims(next_state,0)).float().to(device)
        done_ts = torch.from_numpy(np.array([[int(done)]])).float().to(device)
        
        Q_targets_next = self.qnetwork_target(next_state_ts).detach().max(1)[0].unsqueeze(1)
        Q_targets = reward_ts + (self.gamma * Q_targets_next * (1 - done_ts))
        Q_expected = self.qnetwork_local(state_ts).gather(1, action_ts)

        delta = (Q_targets - Q_expected).detach().cpu().numpy()[0,0]
        return delta
    
    def get_gamma(self):
        return self.gamma
    
    def save_model(self):
        torch.save(self.checkpoint, self.checkpointfile)
        
    def set_lr(self, lr):
        self.lr = lr
        
    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        
        self.qnetwork_local = QNetwork(checkpoint["input_size"],
                                       checkpoint["output_size"],
                                       self.seed,
                                       checkpoint["hidden_layers"])
        self.qnetwork_local.load_state_dict(checkpoint["state_dict"])
        
    def set_uniform_sampling(self):
        """ Set alpha to 0.0 and beta to 1.0 so that the agent
        becomes equivalent to the uniform sampling.
        """
        self.alpha = 0.0
        self.beta = 1.0
        self.memory.set_priority_params(self.alpha, self.beta)

    def parameter_anneal(self):
        self.alpha = max(0.0, self.alpha-ALPHA_ANNEALING)
        self.beta = min(1.0, self.beta+BETA_ANNEALING) 
        self.memory.set_priority_params(self.alpha, self.beta)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha, beta):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size) # Replay buffer
        self.deltas = deque(maxlen=buffer_size) # Buffer to contain delta for each experience
        self.deltas_argsorted = None            # Indices that would sort self.deltas
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)        
    
    def add(self, state, action, reward, next_state, done, delta):
        """Add a new experience to memory, 
        and add corresponding delta to deltas.
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.deltas.append(delta)
    
    def sample(self):
        """Sample a batch of experiences from memory based on 
        priority probabilities.  Corresponding weight and buffer
        index for each experience are also returned.
        """
        idxs = np.random.choice(len(self.memory), BATCH_SIZE, replace=False, p=self.priority_probs)
        
        states = torch.from_numpy(np.vstack([self.memory[self.deltas_argsorted[idx]].state for idx in idxs if self.memory[self.deltas_argsorted[idx]] is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([self.memory[self.deltas_argsorted[idx]].action for idx in idxs if self.memory[self.deltas_argsorted[idx]] is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([self.memory[self.deltas_argsorted[idx]].reward for idx in idxs if self.memory[self.deltas_argsorted[idx]] is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([self.memory[self.deltas_argsorted[idx]].next_state for idx in idxs if self.memory[self.deltas_argsorted[idx]] is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([self.memory[self.deltas_argsorted[idx]].done for idx in idxs if self.memory[self.deltas_argsorted[idx]] is not None]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(np.expand_dims(np.array([self.is_weights[idx] for idx in idxs]),axis=1)).float().to(device) 
        delta_idxs = [self.deltas_argsorted[idx] for idx in idxs]
          
        return (states, actions, rewards, next_states, dones), weights, delta_idxs

    def set_priority_params(self, alpha, beta):
        """ Compute priority probabilities and corresponding weights.
        The rank-based priority probability is used.  Hence, the probability and
        the weight depend only on the size of buffer.
        
        Params
        ======
            alpha (float 0<=alpha<=1): parameter alpha for priority probability
            beta (float 0<=beta<=1) : parameter beta for the importance-sampling weight
        """
        cur_buff_size = len(self.memory)
        p = (1.0 / np.arange(cur_buff_size,0,-1))**alpha
        self.priority_probs = p/p.sum()
        self.is_weights = ((1.0/cur_buff_size) * (1.0/self.priority_probs))**beta
        self.is_weights /= self.is_weights.max()
        
    def argsort_deltas(self):
        """ Argsort the deltas and store the sorted indices in the deltas_argsorted
        """
        self.deltas_argsorted = np.argsort(self.deltas)
        
    def update_deltas(self, deltas, idxs):
        """ Update deltas corresponding to the given indices.
        """
        for idx, delta in zip(idxs, deltas):
#            print("delta {}".format(delta))
            self.deltas[idx] = delta[0]

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)