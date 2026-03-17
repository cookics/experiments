# BALROG Gemini 3 Pro
To replicate Gemini 3 Pro with the naive agent template, use:

```
export GEMINI_API_KEY=YOUR-GEMINI-API-KEY

python3 -m eval \
  agent.type=naive \
  agent.max_image_history=0 \
  agent.max_text_history=16 \
  eval.num_workers=16 \
  client.client_name=gemini \
  client.model_id=gemini-3-pro-preview
```

For more information, visit:

- [Evaluation tutorial](https://github.com/balrog-ai/BALROG/blob/main/docs/evaluation.md)
- [Paper](https://arxiv.org/abs/2411.13543)
- [BALROG](https://balrogai.com)