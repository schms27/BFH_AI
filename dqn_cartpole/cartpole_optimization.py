import optuna

import train_main

def dqn_objective(trial):

    # params to optimize
    lr = trial.suggest_float('learning_rate', 4.5e-4, 6.5e-4, log=True)
    replay_buffer_size = trial.suggest_int('replay_buffer_size', 1000, 25000, log=True )
    replay_buffer_coldrun = trial.suggest_int('replay_buffer_coldrun', 128, 1000, log=True )
    gamma = trial.suggest_float('gamma', 0.98, 0.999, log=True)
    soft_update_tau = trial.suggest_float('soft_update_tau', 0.001, 0.9, log=True)
    eps_min = trial.suggest_float('epsilon_min', 0.01, 0.1, log=True)
    eps_max = trial.suggest_float('epsilon_max', 0.5, 1.0, log=True)
    eps_decay = trial.suggest_float('epsilon_decay', 0.99, 0.999, log=True)
    #update_target_net = trial.suggest_int('update_target_net', 1, 20, log=True)
    #max_t = trial.suggest_int('max_t', 50, 10000, log=True)

    arguments = [
        "--learning_rate", f"{lr}", 
        "--replay_buffer_size", f"{replay_buffer_size}", 
        "--replay_buffer_coldrun", f"{replay_buffer_coldrun}", 
        "--gamma", f"{gamma}",
        "--soft_update_tau", f"{soft_update_tau}",
        "--epsilon_min", f"{eps_min}",
        "--epsilon_max", f"{eps_max}",
        "--epsilon_decay", f"{eps_decay}",
        #"--update_target_net", f"{update_target_net}",
        #"--max_t", f"{max_t}",
        "--show_plot", False,
        "--experiment_name", f"opt_trial_{trial.number}",
        "--n_episodes", "500"
        ]
    
    _, solved_in_i_episodes = train_main.main(arguments, trial)
    
    return solved_in_i_episodes

# we want optuna to minimize the number of episodes used to solve the env
study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
study.optimize(dqn_objective, n_trials=150)

trial = study.best_trial

print(f"Minimal number of eps: {trial.value}")
print(f"Best hyperparameters: {trial.params}")