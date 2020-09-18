from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('quads_num_agents', [6]),
    ('quads_episode_duration', [15.0]),
    ('quads_mode', ['same_goal']),
    ('quads_dist_between_goals', [0.0]),
    ('quads_collision_reward', [0.1]),
    ('encoder_custom', ['quad_multi_encoder']),
])

_experiment = Experiment(
    'quads_pbt_same_goal',
    'python -m algorithms.appo.train_appo --env=quadrotor_multi --train_for_env_steps=1000000000 --algo=APPO '
    '--use_rnn=False --num_workers=72 --num_envs_per_worker=2 --num_policies=8 --ppo_epochs=1 --rollout=128 '
    '--recurrence=1 --batch_size=1024 --learning_rate=0.0001 --pbt_replace_reward_gap=0.1 --pbt_period_env_steps=1000000 '
    '--pbt_start_mutation=20000000 --with_pbt=True --adam_eps=1e-8 --nonlinearity=tanh --exploration_loss_coeff=0.0 '
    '--actor_critic_share_weights=False --policy_initialization=xavier_uniform --adaptive_stddev=False '
    '--hidden_size=64 --with_vtrace=False --max_policy_lag=100000000 --gae_lambda=1.00 --max_grad_norm=0.0 '
    '--ppo_clip_value=5.0',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_multi_pbt_same_goal_v108', experiments=[_experiment])
