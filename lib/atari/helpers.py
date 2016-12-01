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
    else:
      return getattr(self.env, name)

  def reset(self, *args, **kwargs):
    self.env.reset()
    return self.env.ale.getScreenGrayscale()

  def step(self, *args, **kwargs):
    lives_before = self.env.ale.lives()
    _, reward, done, info = self.env.step(*args, **kwargs)
    lives_after = self.env.ale.lives()
    if lives_before > lives_after:
      done = True

    reward = np.clip(reward, -1, 1)

    next_state = self.env.ale.getScreenGrayscale()
    return next_state, reward, done, info

def atari_make_initial_state(state):
  return np.stack([state] * 4, axis=2)

def atari_make_next_state(state, next_state):
  return np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
