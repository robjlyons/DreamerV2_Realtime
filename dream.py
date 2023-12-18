import gym
import dreamerv2.api as dv2

config = dv2.defaults.update({
    'logdir': 'X:/Dreamer_log/logdir/Gaming',
    'time_limit': 27000,
    'action_repeat': 4,
    'steps': 5e7,
    'eval_every': 2.5e5,
    'log_every': 1e4,
    'prefill': 50000,
    'train_every': 16,
    'clip_rewards': 'tanh',
    'model_opt.lr': 2e-4,
    'actor_opt.lr': 4e-5,
    'critic_opt.lr': 1e-4,
    'actor_ent': 1e-3,
    'discount': 0.999,
    'loss_scales.kl': 0.1,
    'loss_scales.discount': 5.0,
    'expl_behavior': 'greedy',
}).parse_flags()

env = gym.make('RetroArch-v0')
dv2.train(env, config)