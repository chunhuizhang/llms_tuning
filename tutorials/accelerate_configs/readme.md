- 设置日志级别：`ACCELERATE_LOG_LEVEL=info` 以及指定 config file `--config_file {CONFIG}`
  
    ```
    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file {CONFIG} examples/research_projects/toxicity/scripts/gpt-j-6b-toxicity.py --log_with wandb
    ```
