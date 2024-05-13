# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:16:54 2024

@author: asus
"""

import os, signal, sys, time

    
#from ai_economist import foundation

from ai_economist import foundation
  
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
from IPython import display

from utils import plotting  

# Define the configuration of the environment that will be built
import numpy as np
from ai_economist.foundation.base.base_component import BaseComponent, component_registry

@component_registry.add
class BuyWidgetFromVirtualStore(BaseComponent):
    name = "BuyWidgetFromVirtualStore"
    required_entities = ["Coin", "Widget"]  # <--- We can now look up "Widget" in the resource registry
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *base_component_args,
        widget_refresh_rate=0.1,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)
        self.widget_refresh_rate = widget_refresh_rate
        self.available_widget_units = 0
        self.widget_price = 5

    def get_additional_state_fields(self, agent_cls_name):
        return {}

    def additional_reset_steps(self):
        self.available_wood_units = 0

    def get_n_actions(self, agent_cls_name):
        if agent_cls_name == "BasicMobileAgent":
            return 1
        return None

    def generate_masks(self, completions=0):
        masks = {}
        for agent in self.world.agents:
            masks[agent.idx] = np.array([
                agent.state["inventory"]["Coin"] >= self.widget_price and self.available_widget_units > 0
            ])

        return masks

    def component_step(self):
        if random.random() < self.widget_refresh_rate: 
            self.available_widget_units += 1

        for agent in self.world.get_random_order_agents():

            action = agent.get_component_action(self.name)

            if action == 0: # NO-OP. Agent is not interacting with this component.
                continue

            if action == 1: # Agent wants to buy. Execute a purchase if possible.
                if self.available_widget_units > 0 and agent.state["inventory"]["Coin"] >= self.widget_price: 
                    agent.state["inventory"]["Coin"] -= self.widget_price
                    agent.state["inventory"]["Widget"] += 1
                    self.available_widget_units -= 1

            else: # We only declared 1 action for this agent type, so action > 1 is an error.
                raise ValueError

    def generate_observations(self):
        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[agent.idx] = {
                "widget_refresh_rate": self.widget_refresh_rate,
                "available_widget_units": self.available_widget_units,
                "widget_price": self.widget_price
            }

        return obs_dict
    

env_config = {
    # ===== SCENARIO CLASS =====
    # Which Scenario class to use: the class's name in the Scenario Registry (foundation.scenarios).
    # The environment object will be an instance of the Scenario class.
    'scenario_name': 'layout_from_file/simple_wood_and_stone',
    
    # ===== COMPONENTS =====
    # Which components to use (specified as list of ("component_name", {component_kwargs}) tuples).
    #   "component_name" refers to the Component class's name in the Component Registry (foundation.components)
    #   {component_kwargs} is a dictionary of kwargs passed to the Component class
    # The order in which components reset, step, and generate obs follows their listed order below.
    'components': [
        # (1) Building houses
        ('Build', {'skill_dist': "pareto", 'payment_max_skill_multiplier': 3}),
        # (2) Trading collectible resources
        ('ContinuousDoubleAuction', {'max_num_orders': 5}),
        # (3) Movement and resource collection
        ('Gather', {}),
        {'BuyWidgetFromVirtualStore': {'widget_refresh_rate': 0.1}},
    ],
    
    # ===== SCENARIO CLASS ARGUMENTS =====
    # (optional) kwargs that are added by the Scenario class (i.e. not defined in BaseEnvironment)
    'env_layout_file': 'quadrant_25x25_20each_30clump.txt',
    'starting_agent_coin': 10,
    'fixed_four_skill_and_loc': True,
    
    # ===== STANDARD ARGUMENTS ======
    # kwargs that are used by every Scenario class (i.e. defined in BaseEnvironment)
    'n_agents': 4,          # Number of non-planner agents (must be > 1)
    'world_size': [25, 25], # [Height, Width] of the env world
    'episode_length': 1000, # Number of timesteps per episode
    
    # In multi-action-mode, the policy selects an action for each action subspace (defined in component code).
    # Otherwise, the policy selects only 1 action.
    'multi_action_mode_agents': False,
    'multi_action_mode_planner': True,
    
    # When flattening observations, concatenate scalar & vector observations before output.
    # Otherwise, return observations with minimal processing.
    'flatten_observations': False,
    # When Flattening masks, concatenate each action subspace mask into a single array.
    # Note: flatten_masks = True is required for masking action logits in the code below.
    'flatten_masks': True,
}
env = foundation.make_env_instance(**env_config)
env.get_agent(0)

# Note: The code for sampling actions (this cell), and playing an episode (below) are general.  
# That is, it doesn't depend on the Scenario and Component classes used in the environment!

def sample_random_action(agent, mask):
    """Sample random UNMASKED action(s) for agent."""
    # Return a list of actions: 1 for each action subspace
    if agent.multi_action_mode:
        split_masks = np.split(mask, agent.action_spaces.cumsum()[:-1])
        return [np.random.choice(np.arange(len(m_)), p=m_/m_.sum()) for m_ in split_masks]

    # Return a single action
    else:
        return np.random.choice(np.arange(agent.action_spaces), p=mask/mask.sum())

def sample_random_actions(env, obs):
    """Samples random UNMASKED actions for each agent in obs."""
        
    actions = {
        a_idx: sample_random_action(env.get_agent(a_idx), a_obs['action_mask'])
        for a_idx, a_obs in obs.items()
    }

    return actions
obs = env.reset()
actions = sample_random_actions(env, obs)
obs, rew, done, info = env.step(actions)
obs.keys()

for key, val in obs['0'].items(): 
    print("{:50} {}".format(key, type(val)))
    
for agent_idx, reward in rew.items(): 
    print("{:2} {:.3f}".format(agent_idx, reward))
    
done
info
#from google.colab.patches import cv2_imshow,cv2
#import matplotlib.pyplot as plt
import os
def do_plot(env, ax, fig):
    """Plots world state during episode sampling."""
    plotting.plot_env_state(env, ax)
    ax.set_aspect('equal')
    display.display(fig)
    display.clear_output(wait=True)

def play_random_episode(env, plot_every=100, do_dense_logging=False):
    """Plays an episode with randomly sampled actions.
    
    Demonstrates gym-style API:
        obs                  <-- env.reset(...)         # Reset
        obs, rew, done, info <-- env.step(actions, ...) # Interaction loop
    
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Reset
    obs = env.reset(force_dense_logging=do_dense_logging)

    # Interaction loop (w/ plotting)
    for t in range(env.episode_length):
        actions = sample_random_actions(env, obs)
        obs, rew, done, info = env.step(actions)

        if ((t+1) % plot_every) == 0:
            do_plot(env, ax, fig)
            #fig.savefig(str(t)+"n.png")

    if ((t+1) % plot_every) != 0:
        do_plot(env, ax, fig)        
        #fig.savefig("not0.png")

#每运行一次生成10张图片，下面又运行一次，共生成20张单张图片
play_random_episode(env, plot_every=100)

# Play another episode. This time, tell the environment to do dense logging
play_random_episode(env, plot_every=100, do_dense_logging=True)

# Grab the dense log from the env
dense_log = env.previous_episode_dense_log

# Show the evolution of the world state from t=0 to t=200，短周期
fig = plotting.vis_world_range(dense_log, t0=0, tN=200, N=5)

#fig.savefig("short.png")
# Show the evolution of the world state over the full episode，长周期
fig = plotting.vis_world_range(dense_log, N=5)

#关键的一步
fig.savefig("long.png")

# Create image file

# Use the "breakdown" tool to visualize the world state, agent-wise quantities, movement, and trading events
plotting.breakdown(dense_log);
    
    
    
    
    
    
    
    
    
    

