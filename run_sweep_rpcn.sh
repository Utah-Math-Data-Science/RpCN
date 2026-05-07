#!/bin/bash

source ~/.bashrc
job_slot_to_gpu=(-1 2 3 4 5 6)


env_parallel --delay 30 --eta -j $((${#job_slot_to_gpu[@]} - 1)) --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}' 'python src/userfm/main.py +experiment=Steer{dynamical_system} steering=RpCN steering/intermediate_reward=FeynmannKacIntermediateRewardExpectedSample steering.chain_count={chain_count} steering.pcn_update_count={update_count} steering.tilt={tilt} steering.penalization_power=0 model.conf={model_alt_id}' ::: $(duckdb runs.sqlite -c "copy (select * from SweepModelTrained cross join SweepSPTUpdateCount cross join SweepSPTChainCount order by dynamical_system desc) to '/dev/stdout'")
