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

# Define default observation lists for convenience.
NONE = ['none']
OTHER = ['other']

# Survival documentation string updated to reflect the night start.
SURVIVIAL_DOC = """
In Survival, the agent has no defined rewards and the episode ends on death or 24 hr of out-of-game time.
The agent begins in a random biome with no inventory at time=13000 (night) and has access to human-level commands.
Currently the agent has access to the cheating smelting command but this will be removed in a future iteration.
This environment most closely represents the open-world objective of vanilla Minecraft, except the episode ends on death.
"""

MS_PER_STEP = 50

class HumanSurvivalNight(SimpleEmbodimentEnvSpec):
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'MineRLSurvivalNight-v0'
        if 'max_episode_steps' not in kwargs:
            kwargs['max_episode_steps'] = 24 * 60 * 60 * 20  # 24 hours * 20hz
        self.episode_len = kwargs['max_episode_steps']
        super().__init__(*args, **kwargs)

    def create_rewardables(self) -> List[Handler]:
        return []

    def create_agent_start(self) -> List[Handler]:
        return []

    def create_agent_handlers(self) -> List[Handler]:
        return []

    def create_server_world_generators(self) -> List[Handler]:
        return [
            handlers.DefaultWorldGenerator(force_reset="true", generator_options="")
        ]

    def create_server_quit_producers(self) -> List[Handler]:
        return [
            handlers.ServerQuitFromTimeUp(
                (self.episode_len * MS_PER_STEP)),
            handlers.ServerQuitWhenAnyAgentFinishes()
        ]

    def create_server_decorators(self) -> List[Handler]:
        return []

    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            # Force the game to start at night (tick 13000) and freeze time.
            handlers.TimeInitialCondition(
                allow_passage_of_time=False,
                start_time=13000
            ),
            handlers.SpawningInitialCondition(
                allow_spawning=True
            )
        ]

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return True

    def is_from_folder(self, folder: str) -> bool:
        return folder == 'none'

    def get_docstring(self):
        return SURVIVIAL_DOC

    def create_mission_handlers(self) -> List[Handler]:
        return []

    def create_observables(self) -> List[Handler]:
        return [
            handlers.POVObservation(self.resolution),
            handlers.FlatInventoryObservation(ALL_ITEMS),
            handlers.TypeObservation('mainhand', NONE + ALL_ITEMS + OTHER),
            handlers.DamageObservation('mainhand'),
            handlers.MaxDamageObservation('mainhand'),
            handlers.ObservationFromCurrentLocation()
        ]

    def create_actionables(self) -> List[Handler]:
        actionables = [
            handlers.KeyboardAction(k, v) for k, v in INVERSE_KEYMAP.items()
        ]
        actionables += [
            handlers.CraftItem(NONE + ALL_ITEMS),
            handlers.CraftItemNearby(NONE + ALL_ITEMS),
            handlers.SmeltItemNearby(NONE + ALL_ITEMS),
            handlers.PlaceBlock(NONE + ALL_ITEMS),
            handlers.EquipItem(NONE + ALL_ITEMS),
            handlers.Camera(),
        ]
        return actionables
