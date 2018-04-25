import os
import pprint
import inspect

import tensorflow as tf

pp = pprint.PrettyPrinter().pprint

def class_vars(obj):
  return {k:v for k, v in inspect.getmembers(obj)
      if not k.startswith('__') and not callable(k)}

class BaseModel(object):
  """Abstract object representing an Reader model."""
  def __init__(self, config):
    self._saver = None
    self.config = config

    try:
      self._attrs = config.__dict__['__flags']
    except:
      self._attrs = class_vars(config)
    pp(self._attrs)

    self.config = config

    for attr in self._attrs:
      name = attr if not attr.startswith('_') else attr[1:]
      setattr(self, name, getattr(self.config, attr))

  def save_model(self, step=None):
    print(" [*] Saving checkpoints...")
    model_name = type(self).__name__

    if not os.path.exists(self.checkpoint_dir):
      os.makedirs(self.checkpoint_dir)
    self.saver.save(self.sess, self.checkpoint_dir, global_step=step)

  def load_model(self):
    print(" [*] Loading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      fname = os.path.join(self.checkpoint_dir, ckpt_name)
      self.saver.restore(self.sess, fname)
      print(" [*] Load SUCCESS: %s" % fname)
      return True
    else:
      print(" [!] Load FAILED: %s" % self.checkpoint_dir)
      return False

  @property
  def checkpoint_dir(self):
    # return os.path.join('checkpoints', self.model_dir)
    # return os.path.join('checkpoints', self.model_dir)
    return 'checkpoints/Freeway-v0/max_delta-1/discount-0.99/min_delta--1/train_frequency-4/screen_width-84/screen_height-84/double_q-False/history_length-4/learning_rate_minimum-0.00025/ep_end-0.1/action_repeat-4/learning_rate_decay-0.96/learning_rate-0.00025/dueling-False/min_reward--1.0/batch_size-32/ep_end_t-1000000/backend-tf/random_start-30/target_q_update_step-10000/env_name-Freeway-v0/learning_rate_decay_step-50000/model-m1/env_type-detail/learn_start-50000.0/max_step-50000000/scale-10000/ep_start-1.0/max_reward-1.0/memory_size-1000000/cnn_format-NCHW/'

  @property
  def model_dir(self):
    model_dir = self.config.env_name
    for k, v in self._attrs.items():
      if not k.startswith('_') and k not in ['display']:
        model_dir += "/%s-%s" % (k, ",".join([str(i) for i in v])
            if type(v) == list else v)
    return model_dir + '/'

  @property
  def saver(self):
    if self._saver == None:
      self._saver = tf.train.Saver(max_to_keep=10)
    return self._saver
