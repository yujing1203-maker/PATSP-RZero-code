#!/bin/bash
model_name=$1
save_name=$2
# GPU-count-aware (was hardcoded 0..7); RZERO_NGPU defaults to 2 for this box.
NG=${RZERO_NGPU:-2}
pids=()
for ((i=0; i<NG; i++)); do
  CUDA_VISIBLE_DEVICES=$i python question_evaluate/evaluate.py --model "$model_name" --suffix $i --save_name "$save_name" &
  pids[$i]=$!
done
for ((i=0; i<NG; i++)); do
  wait ${pids[$i]} 2>/dev/null
done
echo "all evaluate tasks finished"
