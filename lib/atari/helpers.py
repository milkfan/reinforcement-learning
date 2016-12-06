import numpy as np


class AtariEnvWrapper(object):
    """
    Wraps an Atari environment to end an episode when a life is lost.
    """
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        if name == "step":
            return self._step
        elif name == "reset":
            return self._reset
        else:
            return getattr(self.env, name)

    def _reset(self, *args, **kwargs):
        self.env.reset()
        return np.squeeze(self.env.ale.getScreenGrayscale())

    def _step(self, *args, **kwargs):
        lives_before = self.env.ale.lives()
        _, reward, done, info = self.env.step(*args, **kwargs)
        next_state = np.squeeze(self.env.ale.getScreenGrayscale())
        lives_after = self.env.ale.lives()
        if lives_before > lives_after:
            done = True
        return next_state, reward, done, info

def atari_make_initial_state(state):
    return np.stack([state] * 4, axis=2)

def atari_make_next_state(state, next_state):
    return np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
