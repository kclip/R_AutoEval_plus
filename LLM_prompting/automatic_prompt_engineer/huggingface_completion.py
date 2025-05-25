from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, StoppingCriteria, StoppingCriteriaList#, pipeline
import torch
# from accelerate import Accelerator
import os
import gc

class StopOnPunctuation(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer  # Ensure tokenizer is passed to avoid 'loaded_tokenizer' issues

    def __call__(self, input_ids, scores, **kwargs):
        last_token_id = input_ids[0, -1].item()
        # Decode it and strip whitespace
        last_token = self.tokenizer.decode([last_token_id])
        # Check if any of the punctuation marks are present anywhere in the token
        # Check if the last character is a stopping punctuation mark
        return any(char in [".", "!", "?"] for char in last_token)



class HuggingFaceCompletion:
    _net = None
    _tokenizer = None
    _current_model_name = None
    _device= None
    _accelerator = None
    _e = None
    #"cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def create(cls, prompt, loaded_model, loaded_tokenizer, suffix='', model="None", temperature=0.7, max_tokens=50,  top_p=1.0, frequency_penalty=0.0, n=1):
        if suffix != '':
            raise NotImplementedError
        # Load the model and tokenizer only once
        #if loaded_model is None or loaded_tokenizer is None:
        device = 'cuda:0'
        with torch.no_grad():
            inputs = loaded_tokenizer(prompt, truncation=False, padding=True, return_tensors="pt", padding_side="left")
            #device = next(loaded_model.parameters()).device  # Get the device where the model is located
            inputs = {key: value.to(device) for key, value in inputs.items()}
            stopping_criteria = StopOnPunctuation(loaded_tokenizer)
            outputs = loaded_model.generate(
                #inputs.input_ids,
                inputs['input_ids'],
                #inputs_embeds = embeddings,
                attention_mask=inputs["attention_mask"], #.to(torch.bfloat16),
                # position_ids=position_ids_expanded,  # Pass it explicitly
                #max_length=inputs.input_ids.shape[1] + max_tokens,
                max_length=inputs['input_ids'].shape[1] + max_tokens,
                temperature=temperature,
                num_return_sequences=n,
                pad_token_id=loaded_model.config.pad_token_id,
                stopping_criteria=[stopping_criteria],
                do_sample=True,  # Enable temperature sampling
                top_p=top_p,     # Apply top_p sampling
                repetition_penalty=frequency_penalty  # Apply frequency_penalty (repetition penalty)
            )

            #results = [loaded_tokenizer.decode(output.tolist(), skip_special_tokens=True) for output in outputs]
            #results = [loaded_tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
            #print('----------generated outputs')
            results = []
            for output in outputs:
                # Identify where the input prompt ends
                input_length = inputs['input_ids'].shape[1]  # Length of the original input prompt
                generated_tokens = output[input_length:]  # Exclude the input part
                text = loaded_tokenizer.decode(generated_tokens, skip_special_tokens=True)
                #print('---------------------original full generated text', text)
                #text = loaded_tokenizer.decode(output, skip_special_tokens=True)
                # Search from the end of the text for the last punctuation mark
                for idx in range(len(text) - 1, -1, -1):
                    if text[idx] in [".", "!", "?"]:
                        text = text[:idx + 1]  # Keep the text up to the punctuation
                        break
                #print('---------------------after trimming the generated text', text)

                results.append(text)
            # #print('----------postprocessed outputs')
            # #print('results: ', results)
            # del inputs, outputs
            # # del loaded_model
            # gc.collect()
            # # loaded_model = None
            # torch.cuda.empty_cache()
            # torch.cuda.synchronize()
            # Mimicking OpenAI's response structure
            return {
                "choices": [{"text": result} for result in results]#,
                # "usage": {
                #     "prompt_tokens": inputs['input_ids'].numel(),
                #     "completion_tokens": sum(len(loaded_tokenizer.encode(r)) for r in results),
                #     "total_tokens": inputs['input_ids'].numel() + sum(len(loaded_tokenizer.encode(r)) for r in results)
                # }
            }
