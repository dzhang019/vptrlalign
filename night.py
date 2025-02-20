# Copyright (c) 2020 All Rights Reserved
# Author: William H. Guss, Brandon Houghton
from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.handler import Handler
from typing import List

import gym

import minerl.herobraine
import minerl.herobraine.hero.handlers as handlers
from minerl.herobraine.hero.handlers import TranslationHandler
from minerl.herobraine.hero.mc import ALL_ITEMS, INVERSE_KEYMAP
from minerl.herobraine.env_spec import EnvSpec

from collections import OrderedDict

# none = ['none']
# other = ['other']

# The intent of this env_spec is to have all of the basic commands available to the agent for mirroring human demos
SURVIVIAL_DOC = """

In Survival, the agent has no defined rewards and the episode ends on death or 24 hr of out-of-game time.
 
The agent begins in a random biome with no inventory at time=0 and has access to human-level commands. 

Currently the agent has access to the cheating smelting command but this will be removed in a future iteration. 

This environment most closely represents the open-world objective of vanilla Minecraft, except the episode ends on death
"""
MS_PER_STEP = 50
NONE = 'none'
OTHER = 'other'


class HumanSurvivalNight(SimpleEmbodimentEnvSpec):
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'MineRLSurvivalNight-v0'
        kwargs.setdefault('max_episode_steps', 24 * 60 * 60 * 20)  # 24 hours at 20Hz
        super().__init__(*args, **kwargs)

    def create_server_world_generators(self) -> List[Handler]:
        return [
            handlers.DefaultWorldGenerator(
                force_reset=True,
                reset_to_time=13000  # Sync worldgen time with initial conditions
            )
        ]

    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            handlers.TimeInitialCondition(
                allow_passage_of_time=True,
                start_time=13000  # 13000 ticks = night time
            ),
            handlers.SpawningInitialCondition(allow_spawning=True),
            handlers.DifficultyInitialCondition("normal")  # Add explicit difficulty
        ]

    def create_observables(self) -> List[Handler]:
        return [
            handlers.POVObservation(self.resolution),
            handlers.FlatInventoryObservation(ALL_ITEMS),
            handlers.EquippedItemObservation(  # Replace TypeObservation with this
                items=ALL_ITEMS,
                mainhand=True,
                offhand=False,
                armor=False,
                _default="none",
                _other="other"
            ),
            handlers.DamageObservation('mainhand'),
            handlers.MaxDamageObservation('mainhand'),
            handlers.ObservationFromCurrentLocation(),
            handlers.ObservationFromTime()
        ]

    def create_rewardables(self) -> List[Handler]:
        return []

    def create_agent_start(self) -> List[Handler]:
        return []

    def create_agent_handlers(self) -> List[Handler]:
        return []

    def create_server_quit_producers(self) -> List[Handler]:
        return [
            handlers.ServerQuitFromTimeUp(
                (self.episode_len * MS_PER_STEP)),
            handlers.ServerQuitWhenAnyAgentFinishes()
        ]

    def create_server_decorators(self) -> List[Handler]:
        return []


    def determine_success_from_rewards(self, rewards: list) -> bool:
        # All survival experiemnts are a success =)
        return True

    def is_from_folder(self, folder: str) -> bool:
        return folder == 'none'

    def get_docstring(self):
        return SURVIVIAL_DOC

    def create_mission_handlers(self) -> List[Handler]:
        return []

    def create_actionables(self) -> List[Handler]:
        actionables = [
            handlers.KeyboardAction(k, v) for k, v in INVERSE_KEYMAP.items()
        ]
        actionables += [
            handlers.CraftItem(none + ALL_ITEMS),
            handlers.CraftItemNearby(none + ALL_ITEMS),
            handlers.SmeltItemNearby(none + ALL_ITEMS),
            handlers.PlaceBlock(none + ALL_ITEMS),
            handlers.EquipItem(none + ALL_ITEMS),
            handlers.Camera(),
        ]
        return actionables
