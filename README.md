# Curriculum-LLM

Using automated curriculum learning to enhance LLM's RL training process.

## What's this?

- An automated scoring script to label difficulties in dataset
    - For math dataset, we use PASS@N to measure difficulties
    - For logic dataset, we use num of items (e.g. number of people) to measure difficulties
- A automated sampler in [VeRL](https://github.com/volcengine/verl), we base on [Dynamic Sampler](https://github.com/volcengine/verl/pull/631) with modifications


## Install

Same as [VeRL](https://github.com/volcengine/verl)

## Citation

[Dynamic Sampler](https://github.com/volcengine/verl/pull/631)

[Logic-RL](https://github.com/Unakar/Logic-RL)

