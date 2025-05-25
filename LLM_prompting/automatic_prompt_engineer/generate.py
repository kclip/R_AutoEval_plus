from faulthandler import disable
from automatic_prompt_engineer import data, llm


def get_query(prompt_gen_template, demos_template, subsampled_data):
    """
    Returns a query for the prompt generator. A query is the prompt that is sent to the LLM.
    Parameters:
        prompt_gen_template: The template for the prompt generator queries.
        demos_template: The template for the demonstrations.
        subsampled_data: The data to use for the demonstrations.
    Returns:
        A query for the prompt generator.
    """
    inputs, outputs = subsampled_data
    demos = demos_template.fill(subsampled_data)
    return prompt_gen_template.fill(input=inputs[0], output=outputs[0], full_demo=demos)



def auto_reduce_m(fn, queries, n, loaded_model, loaded_tokenizer):
    """Reduces n by half until the function succeeds."""
    try:
        return fn(queries, n=n, loaded_model=loaded_model, loaded_tokenizer=loaded_tokenizer)
    except:
        if n == 1:
            print('lack of memory')
            sdffssdf
        return auto_reduce_m(fn, queries, n, loaded_model, loaded_tokenizer) + auto_reduce_m(fn, queries, n, loaded_model, loaded_tokenizer)

def generate_prompts(prompt_gen_template, demos_template, prompt_gen_data, config, loaded_model, loaded_tokenizer):
    """
    Generates prompts using the prompt generator.
    Parameters:
        prompt_gen_template: The template for the prompt generator queries.
        demos_template: The template for the demonstrations.
        prompt_gen_data: The data to use for prompt generation.
        config: The configuration dictionary.
    Returns:
        A list of prompts.
    """
    queries = []
    # subsampled_data_compl_list = []
    for _ in range(config['num_subsamples']):
        subsampled_data = data.subsample_data(
            prompt_gen_data, config['num_demos'])
        # subsampled_data, subsampled_data_compl = data.subsample_data_for_cross_val(
        #     prompt_gen_data, config['num_demos'])
        queries.append(get_query(prompt_gen_template,
                                 demos_template, subsampled_data))
        # subsampled_data_compl_list.append(subsampled_data_compl)
    # Instantiate the LLM
    model = llm.model_from_config(config['model'], disable_tqdm=False)
    #print('----------------inside generate.py')

    prompts = auto_reduce_m(model.generate_text, queries, n=config['num_prompts_per_subsample'], loaded_model=loaded_model, loaded_tokenizer=loaded_tokenizer)
    # print('generated prompts: ', prompts)
    return prompts

def generate_prompts_for_cross_val(prompt_gen_template, demos_template, prompt_gen_data, config):
    """
    Generates prompts using the prompt generator.
    Parameters:
        prompt_gen_template: The template for the prompt generator queries.
        demos_template: The template for the demonstrations.
        prompt_gen_data: The data to use for prompt generation.
        config: The configuration dictionary.
    Returns:
        A list of prompts.
    """
    # Instantiate the LLM
    model = llm.model_from_config(config['model'], disable_tqdm=False)
    #print('----------------inside generate.py')
    prompts_cv = []
    data_cv = []
    for _ in range(config['num_subsamples']):
        queries = []
        subsampled_data, subsampled_data_compl = data.subsample_data_for_cross_val(
            prompt_gen_data, config['num_demos'])
        # print('subsampled_data', subsampled_data, 'subsampled_data_compl', subsampled_data_compl)
        queries.append(get_query(prompt_gen_template,
                                 demos_template, subsampled_data))
        data_cv.append(subsampled_data_compl)
        prompts = model.generate_text(
            queries, n=config['num_prompts_per_subsample'])
        prompts_cv.append(prompts)
    return prompts_cv, data_cv
