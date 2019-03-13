""" Contains the Episodes for Navigation. """
import random
import torch
import time
import sys
from constants import GOAL_SUCCESS_REWARD, STEP_PENALTY, BASIC_ACTIONS, FAILED_ACTION_PENALTY
from environment import Environment
from utils.net_util import gpuify


class Episode:
    """ Episode for Navigation. """
    def __init__(self, args, gpu_id, rank, strict_done=False):
        super(Episode, self).__init__()

        self._env = None

        self.gpu_id = gpu_id
        self.strict_done = strict_done
        self.task_data = None
        self.glove_embedding = None

        self.seed = args.seed + rank
        random.seed(self.seed)

        with open('./datasets/objects/int_objects.txt') as f:
            int_objects = [s.strip() for s in f.readlines()]
        with open('./datasets/objects/rec_objects.txt') as f:
            rec_objects = [s.strip() for s in f.readlines()]
        
        self.objects = int_objects + rec_objects

        self.actions_list = [{'action':a} for a in BASIC_ACTIONS]
        self.actions_taken = []
        # # information about whether tomato/bowl is found; initial false
        # self.tomato_done = False 
        # self.bowl_done = False

    @property
    def environment(self):
        return self._env

    def state_for_agent(self):
        return self.environment.current_frame

    def step(self, action_as_int):
        action = self.actions_list[action_as_int]
        self.actions_taken.append(action)
        return self.action_step(action)

    def action_step(self, action):
        self.environment.step(action)
        reward, terminal, action_was_successful = self.judge(action)

        return reward, terminal, action_was_successful

    def slow_replay(self, delay=0.2):
        # Reset the episode
        self._env.reset(self.cur_scene, change_seed = False)
        
        for action in self.actions_taken:
            self.action_step(action)
            time.sleep(delay)
    
    def judge(self, action):
        """ Judge the last event. """
        # immediate reward
        reward = STEP_PENALTY 
        done = False
        action_was_successful = self.environment.last_action_success

        # if action is tomato done
        if action['action'] == 'Tomato_Done':
            print('Tomato_Done', action_was_successful)
            objects = self._env.last_event.metadata['objects']
            visible_objects = [o['objectType'] for o in objects if o['visible']]
            if self.target[0] in visible_objects:
                self.tomato_done = True
                reward += GOAL_SUCCESS_REWARD/2
                self.tomato_success = True
            else:
                reward += FAILED_ACTION_PENALTY

        # if action is bowl done
        if action['action'] == 'Bowl_Done':
            print('Bowl_Done', action_was_successful)
            objects = self._env.last_event.metadata['objects']
            visible_objects = [o['objectType'] for o in objects if o['visible']]
            if self.target[1] in visible_objects:
                self.bowl_done = True
                reward += GOAL_SUCCESS_REWARD/2
                self.bowl_success = True
            else:
                reward += FAILED_ACTION_PENALTY

        if not action_was_successful:
            reward += FAILED_ACTION_PENALTY

        # an episode is done only if tomato action is done and bowl action is done
        #if self.tomato_done and self.bowl_done:
        #    done = True
        # an episode is success only if tomato is found and bowl is found
        if self.tomato_success and self.bowl_success:
            #reward *= 2
            self.success = True
            reward += GOAL_SUCCESS_REWARD
            done = True

        return reward, done, [action_was_successful, [self.tomato_done, self.bowl_done]]

    def new_episode(self, args, scene):
        
        if self._env is None:
            if args.arch == 'osx':
                local_executable_path = './datasets/builds/thor-local-OSXIntel64.app/Contents/MacOS/thor-local-OSXIntel64'
            else:
                local_executable_path = './datasets/builds/thor-local-Linux64'
            
            self._env = Environment(
                    grid_size=args.grid_size,
                    fov=args.fov,
                    local_executable_path=local_executable_path,
                    randomize_objects=args.randomize_objects,
                    seed=self.seed)
            self._env.start(scene, self.gpu_id)
        else:
            self._env.reset(scene)

        # two targets: tomato and bowl
        self.target = ['Tomato','Bowl']
        # whether find objects successfully, initial false
        self.success = False
        self.tomato_success = False
        self.bowl_success = False
        self.cur_scene = scene
        self.actions_taken = []
        # whether action object_done has taken, initial false
        self.tomato_done = False 
        self.bowl_done = False
        
        return True
