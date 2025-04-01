from src.model.modeling_llama import LolcatsLlamaModel
import torch
model = LolcatsLlamaModel.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    cache_dir="models",
    ).to("cuda")

print("model loaded")
input_ids = torch.randint(0, 100, (1, 10)).to("cuda")
attn_mask = torch.ones_like(input_ids).to("cuda")
model.forward(
    input_ids=input_ids,
    attention_mask=attn_mask,
    position_ids=torch.arange(10).to("cuda").unsqueeze(0),
    past_key_values=None,
    use_cache=False,
    output_attentions=True,
    output_hidden_states=True)
print("success")