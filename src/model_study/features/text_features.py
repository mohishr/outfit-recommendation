import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_EMBED_DIM = 64

class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, out_dim=TEXT_EMBED_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embed_dim, out_dim)

    def forward(self, token_ids):
        # token_ids: [batch, seq]
        x = self.embedding(token_ids).mean(dim=1) # mean pooling
        return self.fc(x)


def extract_text_feature(token_ids: torch.Tensor, model: SimpleTextEncoder, device=device):
    token_ids = token_ids.to(device)
    model.eval()
    with torch.no_grad():
        emb = model(token_ids)
        return emb.cpu().numpy().squeeze()

if __name__ == "__main__":
    # Example usage
    model = SimpleTextEncoder().to(device)
    sample_token_ids = torch.tensor([[1,2,3,4,5]])  # Example token IDs
    feature = extract_text_feature(sample_token_ids, model)
    print("Extracted text feature shape:", feature)