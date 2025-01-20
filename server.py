from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import GPT2LMHeadModel, GPT2Config
from peft import get_peft_model, LoraConfig, TaskType

# Load GPT-2 model and tokenizer
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

# Define split point
split_layer = 6

class ServerModel(GPT2LMHeadModel):
    def __init__(self, model, split_layer):
        super().__init__(model.config)
        self.server_layers = model.transformer.h[split_layer:]
        self.ln_f = model.transformer.ln_f
        self.lm_head = model.lm_head

    def forward(self, hidden_states=None, labels=None, input_ids=None, **kwargs):
        # Process hidden_states, ignore input_ids if provided by Peft
        if hidden_states is None:
            raise ValueError("hidden_states must be provided")

        for layer in self.server_layers:
            hidden_states = layer(hidden_states)[0]
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"logits": logits.tolist(), "loss": loss.item() if loss is not None else None}


server_model = ServerModel(model, split_layer)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"]
)
server_model = get_peft_model(server_model, lora_config)

app = FastAPI()

class InputData(BaseModel):
    hidden_states: list
    labels: list

@app.post("/predict")
def predict(data: InputData):
    hidden_states = torch.tensor(data.hidden_states)
    labels = torch.tensor(data.labels)
    result = server_model(hidden_states=hidden_states, labels=labels)
    return {"logits": result["logits"], "loss": result["loss"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
