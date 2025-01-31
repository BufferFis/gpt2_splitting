from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

# Load DeepSeek model
model_name = "DeepSeek/DeepSeek-R1"  # Replace with the actual DeepSeek model name
model = AutoModelForCausalLM.from_pretrained(model_name)

split_layer = 6  # Adjust this based on the DeepSeek model architecture

class ServerModel(AutoModelForCausalLM):
    def __init__(self, model, split_layer):
        super().__init__(model.config)
        self.server_layers = model.transformer.h[split_layer:]
        self.ln_f = model.transformer.ln_f
        self.lm_head = model.lm_head

    def forward(self, hidden_states=None, labels=None, input_ids=None, **kwargs):
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
    output = server_model(hidden_states=hidden_states, labels=labels)
    return output