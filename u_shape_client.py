import requests
import torch
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

# Load GPT-2 model and tokenizer
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define split indices for a three-way split
split1 = 2  # Client does layers 0-3 (initial)
split2 = 6  # Server does layers 4-7 (middle), client does layers 8-11 (final)

class ClientModel(torch.nn.Module):
    def __init__(self, model, split1, split2):
        super().__init__()
        # Embeddings remain on the client.
        self.embeddings = model.transformer.wte
        self.position_embeddings = model.transformer.wpe
        
        # Client initial layers (blocks 0 to split1-1)
        self.initial_layers = model.transformer.h[:split1]
        # Client final layers (blocks split2 to end)
        self.final_layers = model.transformer.h[split2:]
        # Final layer norm and lm head remain on client.
        self.ln_f = model.transformer.ln_f
        self.lm_head = model.lm_head

    def forward(self, input_ids):
        # Compute input embeddings.
        inputs_embeds = self.embeddings(input_ids) + self.position_embeddings(
            torch.arange(input_ids.size(1), device=input_ids.device)
        )
        hidden = inputs_embeds
        # Pass through initial client layers.
        for layer in self.initial_layers:
            hidden = layer(hidden)[0]

        # Send hidden states to server for processing of middle layers.
        payload = {"hidden_states": hidden.tolist()}
        response = requests.post("http://127.0.0.1:8000/process", json=payload)
        if response.status_code == 200:
            server_hidden = torch.tensor(response.json()["hidden_states"])
        else:
            raise Exception("Server error: " + response.text)

        # Continue processing on client with final layers.
        hidden = server_hidden
        for layer in self.final_layers:
            hidden = layer(hidden)[0]
        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)
        return logits

client_model = ClientModel(model, split1, split2)

def send_to_server(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    logits = client_model(inputs["input_ids"])
    predicted_token = torch.argmax(logits, dim=-1)
    predicted_text = tokenizer.decode(predicted_token[0])
    print("Predicted:", predicted_text)

if __name__ == "__main__":
    sample_text = "My name is"
    send_to_server(sample_text)
