from gym.envs.registration import register

register(id='replab-v0',
         entry_point='gym_replab.envs:ReplabEnv',
         )
         
