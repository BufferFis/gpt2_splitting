import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from peft import get_peft_model, LoraConfig, TaskType

# Load GPT-2 model and tokenizer
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define split point
split_layer = 6  # Example split layer

class ClientModel(torch.nn.Module):
    def __init__(self, model, split_layer):
        super().__init__()
        self.embeddings = model.transformer.wte
        self.position_embeddings = model.transformer.wpe
        self.client_layers = model.transformer.h[:split_layer]

    def forward(self, input_ids):
        inputs_embeds = self.embeddings(input_ids) + self.position_embeddings(
            torch.arange(input_ids.size(1), device=input_ids.device)
        )
        for layer in self.client_layers:
            inputs_embeds = layer(inputs_embeds)[0]
        return inputs_embeds

class ServerModel(GPT2LMHeadModel):
    def __init__(self, model, split_layer):
        super().__init__(model.config)
        self.server_layers = model.transformer.h[split_layer:]
        self.ln_f = model.transformer.ln_f
        self.lm_head = model.lm_head

    def forward(self, input_ids=None, hidden_states=None, labels=None, **kwargs):
        if hidden_states is None:
            raise ValueError("Expected hidden_states from client, but got None.")
        
        for layer in self.server_layers:
            hidden_states = layer(hidden_states)[0]
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return logits if loss is None else (logits, loss)


# Initialize client and server models
client_model = ClientModel(model, split_layer)
server_model = ServerModel(model, split_layer)

# Apply LoRA to the server model
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"]
)
server_model = get_peft_model(server_model, lora_config)

def train_splitlora(input_text):
    optimizer = torch.optim.AdamW(server_model.parameters(), lr=1e-4)
    inputs = tokenizer(input_text, return_tensors="pt")
    labels = inputs["input_ids"].clone()
    
    # Get the client-side output
    client_output = client_model(inputs["input_ids"])
    
    # Call server model with the correct keyword argument
    logits, loss = server_model(hidden_states=client_output, labels=labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Training step completed. Loss: {loss.item()}")

if __name__ == "__main__":
    sample_text = "WOW"
    train_splitlora(sample_text)