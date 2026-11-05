#!/bin/bash

source ~/.bashrc
job_slot_to_gpu=(-1 0 1 2 3 5 6)


env_parallel -k --eta -j $((${#job_slot_to_gpu[@]} - 1)) --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}' 'python src/userfm/main.py log_constraint_values=true +experiment=Steer{dynamical_system} steering=CREPE steering.level_count={level_count} steering.block_count={block_count} steering.tilt={tilt} steering.penalization_power=0 model.conf={model_alt_id}' ::: $(duckdb runs.sqlite -c "copy (select * from SweepModelTrained where dynamical_system = 'FitzHughNagumo' order by dynamical_system desc) to '/dev/stdout'") ::: level_count 2 4 6 8 ::: block_count 8 12 20
