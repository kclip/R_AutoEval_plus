## Raw Data for the Experiments on R-AutoEval+

### Raw Data (LLM Decisions)
- In order to evaluate each single element of the real-world loss table, one needs to run each model (LLM with different setting, i.e., quantization setting or prompt template) to get the respective decision and compare it with the real-world label. To get the loss table that corresponds to the synthetic data, one needs to run a stronger LLM to get its decision and replace the true label with it. The rest follows the same procedure with the real-world loss table generation. 

- The decisions made by LLMs are saved in the following anonymous link 
```
    https://www.dropbox.com/scl/fi/axri8xib6upnduikop30z/dataset.tar.gz?rlkey=bmrolaa7ikrkq8dltzwcgf2pr&e=2&st=i1xetx7m&dl=0
```
and the corresponding loss tables generations can be found in the folder `/runs_loss_table_gen/` for each task. 

### Generating Raw Data
- The code for generating the raw data can also be found in the folder `/runs_loss_table_gen/`. 
- Generating raw data for LLM quantization task builds upon the code https://github.com/yugjerry/conformal-alignment?tab=readme-ov-file and for LLM prompting task builds upon the code https://github.com/keirp/automatic_prompt_engineer.
