import requests
import torch
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

# Load GPT-2 model and tokenizer
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

split_layer = 6

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

client_model = ClientModel(model, split_layer)

def send_to_server(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    labels = inputs["input_ids"].clone()
    client_output = client_model(inputs["input_ids"])

    payload = {
    "hidden_states": client_output.tolist(),
    "labels": labels.tolist()
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Server Response - Loss: {result['loss']}")
    else:
        print("Error:", response.text)

if __name__ == "__main__":
    sample_text = "WOW"
    send_to_server(sample_text)
