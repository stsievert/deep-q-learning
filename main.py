from time import perf_counter
import gym
from types import SimpleNamespace
from torch import optim

from utils.learn import e_greedy_action
from utils.logger import Logger
from utils.models import ReplayMemory, History
from utils.net import (
    DeepQNetwork,
    Q_targets,
    Q_values,
    save_network,
    copy_network,
    gradient_descent,
)
from utils.processing import phi_map, tuple_to_numpy


def read_flags():
    flags = {
        "floyd": False,
        "data_dir": "./_data",
        "log_dir": None,
        "in_dir": "./_data",
        "log_freq": 1,
        "log_console": True,
        "save_model_freq": 10,
    }
    FLAGS = SimpleNamespace(**flags)
    return FLAGS


# ----------------------------------
FLAGS = read_flags()
# ----------------------------------
# Tranining
env = gym.make("CarRacing-v0")

# Current iteration
step = 0

# Has trained model
has_trained_model = False

# Init training params
params = {
    "num_episodes": 4000,
    "minibatch_size": 32,
    "max_episode_length": int(10e6),  # T
    "memory_size": int(4.5e5),  # N
    "history_size": 4,  # k
    "train_freq": 4,
    "target_update_freq": 10000,  # C: Target nerwork update frequency
    "num_states": len(env.action_space.high),
    "min_steps_train": 50000,
    "render": False,
}

# Initialize Logger
log = Logger(log_dir=FLAGS.log_dir)

# Initialize replay memory D to capacity N
D = ReplayMemory(
    N=params["memory_size"],
    n_states=params["num_states"],
    load_existing=False,
    data_dir=FLAGS.in_dir,
)
skip_fill_memory = D.count > 0

# Initialize action-value function Q with random weights
Q = DeepQNetwork(params["num_states"])
log.network(Q)

# Initialize target action-value function Q^
Q_ = copy_network(Q)

# Init network optimizer
optimizer = optim.Adagrad(Q.parameters())

# Initialize sequence s1 = {x1} and preprocessed sequence phi = phi(s1)
H = History.initial(env)

for ep in range(params["num_episodes"]):

    phi = phi_map(H.get())
    # del phi

    if (ep % FLAGS.save_model_freq) == 0:
        save_network(Q, ep, out_dir=FLAGS.data_dir)

    _start = perf_counter()
    for k in range(params["max_episode_length"]):
        # env.render(mode='human')
        #  if step % 100 == 0:
        #  print('Memory usage: {} (kb)'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

        step += 1
        # Select action a_t for current state s_t
        action, epsilon = e_greedy_action(Q, phi, env, step)
        if step % FLAGS.log_freq == 0:
            log.epsilon(epsilon, step)
        # Execute action a_t in emulator and observe reward r_t and image x_(t+1)
        image, reward, done, _ = env.step(action.tolist())
        if params["render"]:
            env.render()

        # Clip reward to range [-1, 1]
        reward = max(-1.0, min(reward, 1.0))
        # Set s_(t+1) = s_t, a_t, x_(t+1) and preprocess phi_(t+1) =  phi_map( s_(t+1) )
        H.add(image)
        phi_prev, phi = phi, phi_map(H.get())
        # Store transition (phi_t, a_t, r_t, phi_(t+1)) in D
        D.add((phi_prev, action, reward, phi, done))

        should_train_model = skip_fill_memory or (
            (step > params["min_steps_train"])
            and D.can_sample(params["minibatch_size"])
            and (step % params["train_freq"] == 0)
        )

        if should_train_model:
            if not (skip_fill_memory or has_trained_model):
                D.save(params["min_steps_train"])
            has_trained_model = True

            # Sample random minibatch of transitions ( phi_j, a_j, r_j, phi_(j+1)) from D
            phi_mb, a_mb, r_mb, phi_plus1_mb, done_mb = D.sample(
                params["minibatch_size"]
            )

            # Perform a gradient descent step on ( y_j - Q(phi_j, a_j) )^2
            y = Q_targets(phi_plus1_mb, r_mb, done_mb, Q_)
            q_values = Q_values(Q, phi_mb, a_mb)
            q_phi, loss = gradient_descent(y, q_values, optimizer)

            # Log Loss
            if step % (params["train_freq"] * FLAGS.log_freq) == 0:
                log.q_loss(q_phi, loss, step)

            # Reset Q_
            if step % params["target_update_freq"] == 0:
                del Q_
                Q_ = copy_network(Q)

        log.episode(reward)

        # if FLAGS.log_console:
        #     log.display()

        # Restart game if done
        if done:
            H = History.initial(env)
            log.reset_episode()
            break

    # No rendering: ~0.012
    # Rendering: ~0.027
    step_avg_time = (perf_counter() - _start) / (k + 1)
    print(f"Time per step: {step_avg_time}")
