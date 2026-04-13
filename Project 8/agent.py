# External libraries
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from collections import OrderedDict

# Local modules
from environment import PongEnviroment
from utils import parse_args, set_seed, plot_train_rewards, plot_evaluation_rewards

class Policy(nn.Module):
    def __init__(self, state_space: int, action_space: int, hidden_dim: int):
        super().__init__()
        # TODO: Define policy network architecture
        # Read the pytorch documentation on nn.Sequential to learn how to use it        
        self.network = nn.Sequential(OrderedDict([
            # TODO: Fill with the necessary layers
            ('policy_layer_1', nn.Linear(state_space, hidden_dim)),
            ('policy_activation_1', nn.ReLU()),
            ('policy_layer_2', nn.Linear(hidden_dim, hidden_dim)),
            ('policy_activation_2', nn.ReLU()),
            ('policy_layer_3', nn.Linear(hidden_dim, action_space)),
        ]))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        return self.network(state)

class Value(nn.Module):
    def __init__(self, state_space: int, hidden_dim: int):
        super().__init__()
        # TODO: Define policy network architecture
        # Read the pytorch documentation on nn.Sequential to learn how to use it        
        self.network = nn.Sequential(OrderedDict([
            # TODO: Fill with the necessary layers
            ('value_layer_1', nn.Linear(state_space, hidden_dim)),
            ('value_activation_1', nn.ReLU()),
            ('value_layer_2', nn.Linear(hidden_dim, hidden_dim)),
            ('value_activation_2', nn.ReLU()),
            ('value_layer_3', nn.Linear(hidden_dim, 1)),
        ]))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        return self.network(state)


class Agent(nn.Module):
    def __init__(
        self,
        state_space: int,
        action_space: int,
        gamma: float,
        lr: float,
        max_training_steps: int,
    ):
        """
        Initialize the agent.

        Parameters:
            state_space      (tuple) : Size of the state space
            action_space       (int) : Size of the action space
            gamma            (float) : Discount factor
            lr               (float) : Learning rate
            max_training_steps (int) : Max steps in training episodes
        """
        super(Agent, self).__init__()

        # Initialize parameters
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.lr = lr
        self.max_training_steps = max_training_steps

        # TODO: Initialize policy network (actor) 
        # Refer to the handout for the value of hidden_dim
        hidden_dim = 256
        self.policy = Policy(state_space, action_space, hidden_dim)

        # TODO: Initialize value network (critic)
        # Refer to the handout for the value of hidden_dim
        self.value = Value(state_space, hidden_dim)

        # TODO: Initialize optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.lr)

        # TODO: Precalculate cummulative discount factors 
        self.cum_gamma = torch.tensor([self.gamma ** i for i in range(10000)])

    def get_action(self, state: torch.Tensor) -> int:
        """
        Helper function to get the action the agent will take given the state
        of the environment.
        
        IMPORTANT: Should only be used when deploying the agent.

        Parameters:
            state (torch.Tensor) : State encoded as tensor.

        Returns:
            Returns an integer denoting the action the selected action.
        """
        # TODO: Sample action from the policy
        # HINT: Remember how to stop gradient propagation

        with torch.no_grad():
            logits = self.policy(state)

            dist = torch.distributions.Categorical(logits=logits)

            action = dist.sample().item()

        return action
        
    def n_step_returns(self, n: int, rewards: torch.Tensor, next_state_values: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
        """
        Calculates the N-Step-Returns for every timestep

        n_step_returns_t = r_t + r_t+1 * gamma + r_t+2 * gamma ^ 2 + ... + r_t+n-1 * gamma ^ n-1 + V(s_t+n) * gamma ^ n

        Parameters:
            rewards (torch.Tensor): Rewards at timesteps t, t+1, ...
            next_state_values (torch.Tensor): Estimated value for states at timesteps t+1, t+2, ...
            terminated (torch.Tensor): True for the terminal states in next_states_values
        """
        # TODO: Calculate the N-Step-Returns
        # Hint: Use torch.Tensor.cumsum, torch.nn.functional.pad and self.cum_gamma for efficient implementation
        # IMPORTANT: Read the pytorch documentation for these functions carefully, especially for torch.nn.functional.pad

        T = rewards.shape[1]
        n = min(n, T)

        gammas = self.cum_gamma[:T+n]

        discounted = rewards * gammas[:T].view(1, T, 1)

        padded = torch.nn.functional.pad(discounted, (0, 0, 1, n-1))

        cumsum = torch.Tensor.cumsum(padded, dim=1)

        reward = cumsum[:, n:n+T] - cumsum[:, :T]
        reward = reward / gammas[:T].view(1, T, 1)

        i = torch.arange(T) + (n - 1)
        i = torch.clamp(i, max=T-1)

        values = next_state_values[:, i, :]

        mask = terminated[:, i, :]
        values = values * (~mask)

        value = (self.gamma ** n) * values

        return reward + value

    def value_loss(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the loss used to optimize the value network (critic)

        Parameters:
            states        (torch.Tensor): States at timesteps t, t+1, ...
            rewards       (torch.Tensor): Rewards at timesteps t, t+1, ...
            next_states   (torch.Tensor): States at timesteps t+1, t+2, ...
            terminated    (torch.Tensor): True for the terminal states in next_states
        """
        # TODO: Calculate the loss for the value network (critic)
        # Hint: Remember how to stop gradient propagation for the target calculation
        # Refer to the handout for the value of n
        values = self.value(states)

        with torch.no_grad():
            next_values = self.value(next_states)

            targets = self.n_step_returns(10, rewards, next_values, terminated)

        loss = torch.nn.functional.mse_loss(values, targets)

        return loss

    def update_value(self,) -> None:
        """
        Updates the parameters of the value network (critic) by performing
        a step of gradient descent and sets all gradients to zero.
        """
        # TODO: Use the value optimizer to update the parameters
        # TODO: Use the value optimizer to set all the gradients to zero
        self.value_optimizer.step()
        self.value_optimizer.zero_grad()
        


    def policy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the loss used to optimize the policy network (actor)

        Parameters:
            states        (torch.Tensor): States at timesteps t, t+1, ...
            actions       (torch.Tensor): States at timesteps t, t+1, ...
            rewards       (torch.Tensor): Rewards at timesteps t, t+1, ...
            next_states   (torch.Tensor): States at timesteps t+1, t+2, ...
            terminated    (torch.Tensor): True for the terminal states in next_states
        """
        # TODO: Calculate the loss for the policy network (actor)
        # Hint: Remember how to stop gradient propagation for the target calculation
        # Refer to the handout for the value of n
        logits = self.policy(states)

        log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs = log_probs_all.gather(dim=-1, index=actions)

        values = self.value(states)

        with torch.no_grad():
            next = self.value(next_states)
            targets = self.n_step_returns(10, rewards, next, terminated)

        diff = targets - values

        loss = -(log_probs * diff).mean()

        return loss
    
    def update_policy(self) -> None:
        """
        Updates the parameters of the policy network (actor) by performing
        a step of gradient descent and sets all gradients to zero.
        """
        # TODO: Use the policy optimizer to update the parameters
        # TODO: Use the policy optimizer to set all the gradients to zero
        self.policy_optimizer.step()
        self.policy_optimizer.zero_grad()

    def save(self, filename: str) -> None:
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value.state_dict(),
        }
        torch.save(checkpoint, filename)

    def load(self, filename: str) -> None:
        checkpoint = torch.load(filename, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value.load_state_dict(checkpoint["value_state_dict"])

def deploy_agent(
    agent: Agent, env: PongEnviroment,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run the agent in the environment for one episode

    Parameters:
        agent    (Agent) : Reinforcement Learning Agent
        env (Environment): Environment

    Returns:
        state       (torch.FloatTensor) : States at each timestep with shape (N, T, S)
        action      (torch.IntTensor)   : Actions at each timestep with shape (N, T, 1)
        reward      (torch.FloatTensor) : Rewards after each action with shape (N, T, 1)
        new_state   (torch.FloatTensor) : New states after each action with shape (N, T, S)
        terminated  (torch.BoolTensor)  : Whether the new state is a terminal state (N, T, 1)
    """
    # TODO: Reset the environment
    state = env.reset()

    # Hint: Read the environment API in environment.py
    # IMPORTANT: Do not pass a seed to the environment everytime you reset it
    # you should only call this once in main()

    # Start trajectory
    done = False

    states = []
    actions = []
    rewards = []
    next_states = []
    terminateds = []

    while not done:
        # TODO: Get action from agent
        # HINT: Check the agent's methods
        action = agent.get_action(state)
        # TODO: Take step in environment
        new_state, reward, terminated, truncated = env.step(action)

        # TODO: Store trajectory step
        states.append(state)
        actions.append(torch.tensor([action], dtype=torch.int64))
        rewards.append(torch.tensor([reward], dtype=torch.float32))
        next_states.append(new_state)
        terminateds.append(torch.tensor([terminated], dtype=torch.bool))
        # TODO: Stop the trajectory if the episode got truncated or terminated
        # Hint: Read the environment.py documentation about truncated
        if terminated or truncated:
            done = True

        state = new_state

    # IMPORTANT: Check the shapes and data types for the return values in the
    # docstring
        
    states = torch.stack(states).unsqueeze(0) 
    actions = torch.stack(actions).unsqueeze(0)
    rewards = torch.stack(rewards).unsqueeze(0)
    next_states = torch.stack(next_states).unsqueeze(0)
    terminateds = torch.stack(terminateds).unsqueeze(0)

    return states, actions, rewards, next_states, terminateds


def main():
    args = parse_args()

    if args.eval_only == False:
        set_seed(10301) # DON'T DELETE THIS

        # TODO: Initialize environment
        env = PongEnviroment(max_steps=None, record=False)

        # Set random seed (DON'T DELETE THIS)
        env.reset(seed=10301)

        state_size = sum(env.observation_space.shape)
        action_size = int(env.action_space.n)

        # TODO: Initialize agent
        agent = Agent(state_size, action_size, args.gamma, args.lr, args.max_steps)

        # Train the agent
        train_rewards_list = []
        eval_rewards_list = []
        for episode in tqdm(
            range(args.train_episodes), "Training episodes", leave=False
        ):
            # ============ Training ===========
            # TODO: Deploy the current policy to get a new trajectory
            states, actions, rewards, new_states, terminated = deploy_agent(agent, env)

            # Store training metrics
            train_mean_return = rewards.sum().item()
            train_rewards_list.append(train_mean_return)

            # TODO: Calculate the loss for the policy and value
            # Hint: Check the agent's methods
            value_loss = agent.value_loss(states, rewards, new_states, terminated)
            policy_loss = agent.policy_loss(states, actions, rewards, new_states, terminated)

            # Scale losses and get the gradients
            value_loss = value_loss / args.batch_size
            policy_loss = policy_loss / args.batch_size
            value_loss.backward()
            policy_loss.backward()

            if (episode + 1) % args.batch_size == 0:
                # TODO: Update the policy and value functions
                agent.update_policy()
                agent.update_value()

            # ============ Evaluation ===========
            if (episode + 1) % args.eval_every == 0:
                eval_mean_undiscounted_return = 0
                for eval_episode in range(args.eval_episodes):
                    # TODO: Deplay the agent
                    states, actions, rewards, new_states, terminated = deploy_agent(agent, env)

                    # Store evaluation metrics
                    T = rewards.shape[2]
                    eval_undiscounted_return = rewards.sum().item()
                    eval_mean_undiscounted_return += eval_undiscounted_return / args.eval_episodes
                eval_rewards_list.append(eval_mean_undiscounted_return)

            # Store network checkpoints
            if episode + 1 % args.store_every == 0:
                print("Storing checkpoint")
                agent.save("checkpoint.pth")
        
        # Store final network
        agent.save("checkpoint.pth")

        # Plot returns and moving average vs episodes
        train_rewards_array = np.array(train_rewards_list)
        eval_rewards_array = np.array(eval_rewards_list)
        plot_train_rewards(train_rewards_array)
        plot_evaluation_rewards(eval_rewards_array, args.eval_every)

    else:
        # TODO: Initialize environment with record = True
        env = PongEnviroment(max_steps=None, record=True)

        state_size = sum(env.observation_space.shape)
        action_size = int(env.action_space.n)
        
        # TODO: Initialize agent
        agent = Agent(state_size, action_size, args.gamma, args.lr, args.max_steps)

        # Load networks
        agent.load("checkpoint.pth")

        # Evaluate the agent
        N = 20
        trajectory_indices = torch.arange(0, N)
        rewards_list = []
        for i in range(N):
            # TODO: Deploy the agent to get a trajectory
            states, actions, rewards, new_states, terminated = deploy_agent(agent, env)
            rewards_list.append(rewards)
        env.close()

        # Get the total reward for each trajectory
        rewards = torch.tensor([rewards.sum() for rewards in rewards_list])
        # Get the length of each trajectory
        steps = torch.tensor([rewards.shape[1] for rewards in rewards_list])

        # Identify the longest winning trajectory
        sorted_steps, sort_indices = steps.sort()
        rewards = rewards[sort_indices]
        trajectory_indices = trajectory_indices[sort_indices]
        longest_winning_trajectory = trajectory_indices[rewards > 0][-1].item()
        print(f"Longest winning trajectory: {longest_winning_trajectory}")

if __name__ == "__main__":
    main()
