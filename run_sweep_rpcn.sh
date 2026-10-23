#!/bin/bash

source ~/.bashrc
job_slot_to_gpu=(-1 5 6)


env_parallel -k --eta -j $((${#job_slot_to_gpu[@]} - 1)) --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}' 'python src/userfm/main.py log_constraint_values=true +experiment=Steer{dynamical_system} steering=RpCN steering/intermediate_reward=FeynmanKacIntermediateRewardExpectedSample steering.chain_count={chain_count} steering.pcn_update_count={update_count} steering.tilt={tilt} steering.penalization_power=0 model.conf={model_alt_id}' ::: $(duckdb runs.sqlite -c "copy (select * from SweepModelTrained order by dynamical_system desc) to '/dev/stdout'") ::: chain_count 2 4 6 8 ::: update_count 1 2 4
