import random
import optuna
import numpy as np

## change default to "selector" for that parameter
## there are specifications for what type should be there, see that and do final design. FOr exxample --> learning_rate (Union[float, Callable[[float], float]])
## Fix TENSORBOARD_LOG
## Generate a number/code that is unique to each hyperparmeter configuration



# import stable_baselines3.common.schedules as schedules



#####################################################################
##             LEARNING RATE SCHEDULER (learning_rate)             ##
#####################################################################
# Currently random, make it categorical and pass parameters using dictionary properly

class Learning_Rate_Scheduler:
    # remove this default function later
    # Make this categorical
    # send params in __init__ for all class's functions --> def __init__(self, i, params):
    # always call through 
    

    def __init__(self):
        self.low = 1e-5
        self.high = 1e-2
        self.log = True
        #do this for all
    
    def default(self):
        return random.random() / 1000

    def opt(self, trial):
        return trial.suggest_float("learning_rate", self.low, self.high, log=self.log)

    def step_decay(optimizer, initial_lr=0.001, drop_rate=0.1, epochs_drop=10):
        return
        scheduler = lr_scheduler.StepLR(optimizer, step_size=epochs_drop, gamma=drop_rate)
        return scheduler

    def exp_decay(optimizer, initial_lr=0.001, decay_rate=0.96):
        return
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        return scheduler

    def cosine_anneal(optimizer, initial_lr=0.001, epochs=100):
        return
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        return scheduler

    def reduce_on_plateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False):
        return
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose)
        return scheduler

    def cyclical_lr(optimizer, step_size=2000, base_lr=1e-3, max_lr=6e-3, mode='triangular', gamma=1.):
        return
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size,
                                        mode=mode, gamma=gamma)
        return scheduler

    def warmup_lr(optimizer, factor=10, warmup_epochs=5):
        return
        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                return factor
            else:
                return 1.0
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler

    def one_cycle_lr(optimizer, num_steps, lr_range=(1e-4, 1e-2), momentum_range=(0.85, 0.95)):
        return
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr_range[1], total_steps=num_steps,
                                            pct_start=0.3, anneal_strategy='cos', cycle_momentum=True,
                                            base_momentum=momentum_range[0], max_momentum=momentum_range[1])
        return scheduler

    def constant_schedule(self, lr=3e-4):
        return schedules.constant_schedule(3e-4)

    def linear_schedule(self, initial_value=1e6, final_value=3e-4, n_timsteps=1e-4):
        return schedules.linear_schedule(initial_value, final_value, n_timsteps)

    def piecewise_schedule(self, schedule_pieces):
        return schedules.piecewise_schedule(schedule_pieces)

    def cosine_schedule(self, initial_lr=3e-4, final_lr=1e-4, total_timesteps=1e6): # check if this cosine annealing
        return schedules.cosine_schedule(initial_lr, final_lr, total_timesteps)

    def linear_warmup_schedule(self):
        warmup_timesteps = 1e5
        learning_rate_schedule = schedules.linear_schedule(9e5, 0, 3e-4)
        learning_rate = schedules.linear_warmup(learning_rate_schedule, warmup_timesteps, 0)
        return learning_rate

    def combined_scheduler(self):
        pass



#####################################################################
##                        POLICY(policy)                           ##
#####################################################################
# Categorical
class PolicySelector:
    def __init__(self):
        pass

    def opt(self, trial):
        return "MlpPolicy"
        
    def default(self):
        return self.MlpPolicy()

    def MlpPolicy(self):
        return "MlpPolicy"
    
    def CnnPolicy(self):
        return "CnnPolicy"

    def MultiInputPolicy(self):
        return "MultiInputPolicy"
    

#####################################################################
##                  UPDATE PER STEPS (n_steps)                     ##
#####################################################################
#Refine the search space --> dont just do 2^i in between also there might be peak

class StepsPerUpdate:
    
    def __init__(self):
        self.max_size = 2048

        pass

    def opt(self, trial):
        return 2 ** trial.suggest_int("n_steps", 10, 12)

    def default(self):
        return random.randint(1, self.max_size)

#####################################################################
##                      BATCH SIZE (batch_size)                    ##
#####################################################################

class BatchSize:
    def __init__(self):
        self.max_size = 64
        pass

    def opt(self, trial):
        return 2 ** trial.suggest_int("batch_size", 5, 8)
    
    def default(self):
        self.max_size=26
        return random.randint(1, self.max_size)

#####################################################################
##    No. OF EPOCHS WHEN OPTIMIZING SURROGATE LOSS (n_epochs)      ##
#####################################################################
# Higher the better

class NoOfEpochs:
    def __init__(self):
        self.max_size = 10
        pass

    def opt(self, trial):
        return trial.suggest_int("n_epochs", 5, 10)
    
    def default(self):
        return random.randint(1, self.max_size)


#####################################################################
##                      DISCOUNT FACTOR (gamma)                    ##
#####################################################################

class DiscountFactor:
    def __init__(self):
        self.max_size = 10
    
    def opt(self, trial):
        return trial.suggest_float("gamma", 0.9, 0.999)
    
    def default(self):
        return random.random()

#####################################################################
##    BIAS-VARIANCE TRADEOFF for GENERALIZED ADVANTAGE ESTIMATOR   ##
#####################################################################

class BiasVarianceTradeoff:
    def __init__(self):
        pass

    def opt(self, trial):
        return trial.suggest_float("bvtradeoff", 0.9, 0.99)
    
    def default(self):
        return random.random()

#####################################################################
##                  CLIPPING FUNCTION (clip_range)                 ##
#####################################################################
# Can be a function of progress

class ClipRange:
    def __init__(self):
        pass

    def opt(self, trial):
        return trial.suggest_float("clip_range", 0.1, 0.3)
    
    def default(self):
        return random.random()


#####################################################################
##       CLIPPING RANGE for Value Function (clip_range_vf)         ##
#####################################################################

class ClipRangeVF:
    def __init__(self):
        pass

    def opt(self, trial):
        return trial.suggest_float("clip_range_vf", 0.1, 0.3)
    
    def default(self):
        return random.random()


#####################################################################
##             NORMALIZE ADVANTAGE(normalize_advantage)            ##
#####################################################################

class NormalizeAdvantage:
    def __init__(self):
        pass
    
    def opt(self, trial):
        return False

    def default(self):
        return True


#####################################################################
##                  ENTROPY COEFFICIENT (ent_coef)                 ##
#####################################################################

class EntropyCoefficient:
    def __init__(self):
        pass

    def opt(self, trial):
        return trial.suggest_float("ent_coef", 0, 0.01)
    
    def default(self):
        return 0


#####################################################################
##            VALUE FUNCTION COEFFICIENT (vf_coef)                 ##
#####################################################################

class ValueFunctionCoefficient:
    def __init__(self):
        pass

    def opt(self, trial):
        return trial.suggest_float("vf_coef", 0.25, 0.75)
    
    def default(self):
        return 0.5

#####################################################################
##       MAXIMUM VALUE for Gradient Clipping (max_grad_norm)       ##
#####################################################################

class MaxGradNorm:
    def __init__(self):
        pass

    def opt(self, trial):
        return trial.suggest_float("max_grad_norm", 0.3, 1)
    
    def default(self):
        return 0.5


#####################################################################
##        USE ? Generalized State Dependent Exploration            ##
#####################################################################

class BoolGeneralizedStateDependentExploration:
    def __init__(self):
        pass

    def opt(self, trial):
        return False
    
    def default(self):
        return False


#####################################################################
##             SAMPLE NEW NOISE MATRIX(sde_sample_freq)            ##
#####################################################################

class SDESampleFrequency:
    def __init__(self):
        pass

    def opt(self, trial):
        return -1
    
    def default(self):
        return -1


#####################################################################
##                LIMIT KL-Divergence (target_kl)                  ##
#####################################################################

class TargetKL:
    def __init__(self):
        pass

    def opt(self, trial):
        return trial.suggest_float("target_kl", 0.01, 0.05)
    
    def default(self):
        return None


#####################################################################
##                      TENSORBOARD_LOG                            ##
#####################################################################

# class TensorBoardLog:
#     def __init__(self):
#         pass

#     def opt(self, trial):
#         return 
    
#     def default(self):
#         return None


#####################################################################
##                   POLICY_KWARGS (policy_kwargs)                 ##
#####################################################################

class PolicyKWargs:
    def __init__(self):
        pass

    def opt(self, trial):
        return None
    
    def default(self):
        return None


#####################################################################
##                           SEED (seed)                           ##
#####################################################################

class Seed:
    def __init__(self):
        pass

    def opt(self, trial):
        return trial.suggest_int("seed", 1, 10)
    
    def default(self):
        return None


all_policy = [
    "MlpPolicy",
    "CnnPolicy",
    "MultiInputPolicy",
]

# ~!@ environment is fixed {maybe we can find stronger relations between hyperparameters across other environments}
    # Very important idea


all_classes_in_this_file = [
    'BatchSize',
    'BiasVarianceTradeoff',
    'BoolGeneralizedStateDependentExploration',
    'ClipRange',
     'ClipRangeVF',
      'DiscountFactor',
       'EntropyCoefficient',
        'Learning_Rate_Scheduler',
         'MaxGradNorm',
          'MegaHandler',
           'NoOfEpochs',
            'NormalizeAdvantage',
             'PolicyKWargs',
              'PolicySelector',
               'SDESampleFrequency',
                'Seed',
                 'StepsPerUpdate',
                  'TargetKL',
                   'TensorBoardLog',
                    'ValueFunctionCoefficient'
]

def request_next_HypConfig(env, tb_logs):
    hyps = dict()

    hyps["policy"] = PolicySelector().default()
    hyps["env"] = env
    hyps["learning_rate"] = Learning_Rate_Scheduler().default()
    hyps["n_steps"] = StepsPerUpdate().default()
    hyps["batch_size"] = BatchSize().default()
    hyps["n_epochs"] = NoOfEpochs().default()
    hyps["gamma"] = DiscountFactor().default()
    hyps["gae_lambda"] = BiasVarianceTradeoff().default()
    hyps["clip_range"] = ClipRange().default()
    hyps["clip_range_vf"] = ClipRangeVF().default()
    hyps["normalize_advantage"] = NormalizeAdvantage().default()
    hyps["ent_coef"] = EntropyCoefficient().default()
    hyps["vf_coef"] = ValueFunctionCoefficient().default()
    hyps["max_grad_norm"] = MaxGradNorm().default()
    hyps["use_sde"] = BoolGeneralizedStateDependentExploration().default()
    hyps["sde_sample_freq"] = SDESampleFrequency().default()
    hyps["target_kl"] = TargetKL().default()
    hyps["tensorboard_log"] = tb_logs
    hyps["policy_kwargs"] = PolicyKWargs().default()
    hyps["verbose"] = 0
    hyps["seed"] = Seed().default()
    hyps["device"] = 'auto'
    hyps["_init_setup_model"] = True

    return hyps





#####################################################################
##            MEGA HANDLER (handler of all classes below)          ##
#####################################################################
class HypRequestHandler:
    def __init__(slef):
        pass

    def optuna_next_sample(self, trial, env, tb_logs):
        hyps = dict()

        hyps["policy"] = PolicySelector().opt(trial)
        hyps["env"] = env
        hyps["learning_rate"] = Learning_Rate_Scheduler().opt(trial)
        hyps["n_steps"] = StepsPerUpdate().opt(trial)
        hyps["batch_size"] = BatchSize().opt(trial)
        hyps["n_epochs"] = NoOfEpochs().opt(trial)
        hyps["gamma"] = DiscountFactor().opt(trial)
        hyps["gae_lambda"] = BiasVarianceTradeoff().opt(trial)
        hyps["clip_range"] = ClipRange().opt(trial)
        hyps["clip_range_vf"] = ClipRangeVF().opt(trial)
        hyps["normalize_advantage"] = NormalizeAdvantage().opt(trial)
        hyps["ent_coef"] = EntropyCoefficient().opt(trial)
        hyps["vf_coef"] = ValueFunctionCoefficient().opt(trial)
        hyps["max_grad_norm"] = MaxGradNorm().opt(trial)
        hyps["use_sde"] = BoolGeneralizedStateDependentExploration().opt(trial)
        hyps["sde_sample_freq"] = SDESampleFrequency().opt(trial)
        hyps["target_kl"] = TargetKL().opt(trial)
        hyps["tensorboard_log"] = tb_logs
        hyps["policy_kwargs"] = PolicyKWargs().opt(trial)
        hyps["verbose"] = 0
        hyps["seed"] = 1
        hyps["device"] = 'auto'
        hyps["_init_setup_model"] = True

        return hyps