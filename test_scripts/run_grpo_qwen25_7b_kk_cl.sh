set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

DATASET_PATH=$HOME/curriculum-llm/data/kk-difficulty

nohup python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATASET_PATH/train.parquet \
    data.val_files=$DATASET_PATH/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=400 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.use_dynamic_sampler=True \
    data.init_level=0 \
    data.mix_harder_samples=0.0 \
    data.smooth_window=3 \
    data.threshold_for_next_level=0.7 \
    data.threshold_for_prev_level=0.3 \
    data.seed=42 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='grpo_qwen2.5_7b_curriculum' \
    trainer.experiment_name='grpo_qwen2.5_7b_kk_curriculum' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=30 2>&1 | tee grpo_qwen2.5_7b_kk_curriculum.log &
