# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:04:24 2024

@author: asus
"""


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
    