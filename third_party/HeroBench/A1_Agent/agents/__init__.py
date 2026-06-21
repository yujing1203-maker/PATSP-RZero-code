import os

def load_prompt(prompt_name):
    prompt_name = prompt_name.lower()
    selfpath = os.path.dirname(os.path.realpath(__file__))+'/prompts/'
    path = None
    if prompt_name == 'critic-1':
        path = selfpath + 'critic-1.txt'
    if prompt_name == 'critic-2':
        path = selfpath + 'critic-2.txt'
    if prompt_name == 'action_hl_decompose':
        path = selfpath + 'action_hl_decompose.txt'
    if prompt_name == 'action_ll_decompose':
        path = selfpath + 'action_ll_decompose.txt'

    raw_prompt = ''
    with open(path, 'r') as f:
        raw_prompt = f.read()
    return raw_prompt

def load_prompt2(prompt_name, prompts_version = 1):
    prompt_name = prompt_name.lower()
    base_path = os.path.dirname(os.path.realpath(__file__))+'/prompts/'
    prompt_dir_path = ''
    if 'system' in prompt_name:
        prompt_dir_path = base_path+f'system_prompts_{prompts_version}/'
    else:
        prompt_dir_path = base_path+f'prompts_{prompts_version}/'
    try:
        path = prompt_dir_path+prompt_name
        with open(path, 'r') as f:
            raw_prompt = f.read()
            return raw_prompt

    except FileNotFoundError:
        raise FileNotFoundError(f'Prompt {prompt_name} not found')
