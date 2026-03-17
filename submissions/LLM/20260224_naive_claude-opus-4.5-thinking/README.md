# BALROG Claude 4.5 Opus Thinking
To replicate Claude 4.5 Opus Thinking with the naive agent template, use:

```
export AWS_ACCESS_KEY_ID="YOUR-ACCESS-KEY-ID"
export AWS_SECRET_ACCESS_KEY="YOUR-SECRET-ACCESS-KEY"
export AWS_REGION="YOUR-REGION"

python3 -m eval \
  agent.type=naive \
  agent.max_image_history=0 \
  agent.max_text_history=16 \
  eval.num_workers=16 \
  client.client_name=aws-bedrock \
  client.model_id=global.anthropic.claude-opus-4-5-20251101-v1:0 \
  client.generate_kwargs.thinking_budget=1024 \
  client.base_url=null
```

For more information, visit:

- [Evaluation tutorial](https://github.com/balrog-ai/BALROG/blob/main/docs/evaluation.md)
- [Paper](https://arxiv.org/abs/2411.13543)
- [BALROG](https://balrogai.com)