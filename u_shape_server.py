from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import GPT2LMHeadModel, GPT2Config
from peft import get_peft_model, LoraConfig, TaskType

# Load GPT-2 model and configuration
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

split1 = 4
split2 = 8

class ServerModel(torch.nn.Module):
    def __init__(self, model, split1, split2):
        super().__init__()
        self.config = model.config
        self.middle_layers = model.transformer.h[split1:split2]

    def forward(self, hidden_states=None, **kwargs):
        if hidden_states is None:
            hidden_states = kwargs.get("hidden_states", None)
        if hidden_states is None:
            raise ValueError("hidden_states is required for processing.")
        for layer in self.middle_layers:
            hidden_states = layer(hidden_states)[0]
        return {"hidden_states": hidden_states.tolist()}
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

server_model = ServerModel(model, split1, split2)

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

@app.post("/process")
def process(data: InputData):
    hidden_states = torch.tensor(data.hidden_states)
    # Pass hidden_states as a keyword argument
    result = server_model(hidden_states=hidden_states)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)