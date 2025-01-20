from IPython import get_ipython
from IPython.display import display
# %%
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from peft import get_peft_model, LoraConfig, TaskType

# Load GPT-2 model and tokenizer
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Split model into client and server parts by modifying the forward function
class SplitGPT2(GPT2LMHeadModel):
    def __init__(self, model, split_layer):
        super().__init__(model.config)
        self.client_layers = model.transformer.h[:split_layer]
        self.server_layers = model.transformer.h[split_layer:]
        self.embeddings = model.transformer.wte
        self.position_embeddings = model.transformer.wpe
        self.ln_f = model.transformer.ln_f
        self.lm_head = model.lm_head

    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        token_type_ids=None, 
        position_ids=None, 
        inputs_embeds=None, 
        past_key_values=None, 
        use_cache=False, 
        output_attentions=False, 
        output_hidden_states=False, 
        return_dict=False, 
        labels=None  # Added labels argument
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids) + self.position_embeddings(
                torch.arange(input_ids.size(1), device=input_ids.device)
            )

        # Process through client-side layers
        for layer in self.client_layers:
            # Modify this line to only get the hidden states
            # Assuming the layer returns a tuple (hidden_states, ...)
            inputs_embeds = layer(inputs_embeds)[0]  

        # Simulate sending activations to server
        for layer in self.server_layers:
            # Modify this line to only get the hidden states
            # Assuming the layer returns a tuple (hidden_states, ...)
            inputs_embeds = layer(inputs_embeds)[0]  

        inputs_embeds = self.ln_f(inputs_embeds)
        logits = self.lm_head(inputs_embeds)

        # If labels are provided, calculate loss
        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        if return_dict:
            return {"logits": logits, "loss": loss}
        return logits if loss is None else (logits, loss)


        # Define split point
split_layer = 6  # Example split layer

# Create the split model
split_model = SplitGPT2(model, split_layer)

# Apply LoRA for parameter-efficient fine-tuning
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # GPT-2 is a causal LM
    r=8,                            # Low-rank approximation size
    lora_alpha=32,                   # Scaling factor
    lora_dropout=0.1,                 # Dropout for regularization
    target_modules=["c_attn", "c_proj"]  # Target GPT-2 attention layers
)

split_model = get_peft_model(split_model, lora_config)

# Simulate client-server interaction
def client_forward(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        client_activations = split_model(**inputs)
    return client_activations

def train_splitlora(input_text):
    optimizer = torch.optim.AdamW(split_model.parameters(), lr=1e-4)
    inputs = tokenizer(input_text, return_tensors="pt")
    labels = inputs["input_ids"].clone()

    # Pass both input_ids and labels, and set return_dict=True
    outputs = split_model(input_ids=inputs["input_ids"], labels=labels, return_dict=True)  
    loss = outputs["loss"]  # Retrieve loss from the output dictionary

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Training step completed. Loss: {loss.item()}")
# Simulate training locally
# %%
if __name__ == "__main__":
    sample_text = "WOW"
    train_splitlora(sample_text)