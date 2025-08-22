# BALROG GPT-5 Minimal Thinking
To replicate GPT-5 minimal thinking with the naive agent template, use:

```
export OPENAI_API_KEY=YOUR-OPENAI-API-KEY

python3 -m eval \
  agent.type=naive \
  agent.max_image_history=0 \
  agent.max_text_history=16 \
  eval.num_workers=16 \
  client.client_name=openai \
  client.model_id=gpt-5-2025-08-07
```

The GPT-5 minimal thinking is as close as possible to the base model with no reasoning.

For more information, visit:

- [Evaluation tutorial](https://github.com/balrog-ai/BALROG/blob/main/docs/evaluation.md)
- [Paper](https://arxiv.org/abs/2411.13543)
- [BALROG](https://balrogai.com)