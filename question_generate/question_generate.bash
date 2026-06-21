# load the model name from the command line
model_name=$1
num_samples=$2
save_name=$3
export VLLM_DISABLE_COMPILE_CACHE=1
# GPU-count-aware (was hardcoded 0..7); RZERO_NGPU defaults to 2 for this box.
NG=${RZERO_NGPU:-2}
for ((i=0; i<NG; i++)); do
  CUDA_VISIBLE_DEVICES=$i python question_generate/question_generate.py --model "$model_name" --suffix $i --num_samples "$num_samples" --save_name "$save_name" &
done
wait
