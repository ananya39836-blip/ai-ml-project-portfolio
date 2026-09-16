import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
# -----------------------------
# Two-Tower Model with Transformer Encoders
# -----------------------------
class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, item_dim, embed_dim):
        super().__init__()
        
        # User tower (transformer encoder for sequence features)
        self.user_encoder = nn.Sequential(
            nn.Linear(user_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
        
        # Item tower (transformer encoder for metadata features)
        self.item_encoder = nn.Sequential(
            nn.Linear(item_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, user_features, item_features):
        user_emb = self.user_encoder(user_features)
        item_emb = self.item_encoder(item_features)
        
        # Normalize for cosine similarity
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)
        return user_emb, item_emb

# -----------------------------
# Contrastive Loss (InfoNCE style)
# -----------------------------
def contrastive_loss(user_emb, item_emb, temperature=0.05):
    logits = torch.matmul(user_emb, item_emb.T) / temperature
    labels = torch.arange(user_emb.size(0)).to(user_emb.device)
    return F.cross_entropy(logits, labels)

# -----------------------------
# Training Loop (simplified)
# -----------------------------
user_dim, item_dim, embed_dim = 64, 128, 64
model = TwoTowerModel(user_dim, item_dim, embed_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    user_features = torch.rand(32, user_dim)
    item_features = torch.rand(32, item_dim)
    
    user_emb, item_emb = model(user_features, item_features)
    loss = contrastive_loss(user_emb, item_emb)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# -----------------------------
# Real-time Retrieval with FAISS
# -----------------------------
# Precompute item embeddings
item_features = torch.rand(10000, item_dim)
with torch.no_grad():
    _, item_embs = model(torch.rand(1, user_dim), item_features)

item_embs_np = item_embs.cpu().numpy()

# Build FAISS index
index = faiss.IndexFlatIP(embed_dim)  # inner product similarity
index.add(item_embs_np)

# Query with a new user
user_features = torch.rand(1, user_dim)
with torch.no_grad():
    user_emb, _ = model(user_features, torch.rand(1, item_dim))

user_emb_np = user_emb.cpu().numpy()
D, I = index.search(user_emb_np, k=5)  # top-5 items
print("Top-5 retrieved item IDs:", I[0])
class UserTower(nn.Module):
    def __init__(self, embed_dim, seq_len, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, user_seq):
        x = self.embedding(user_seq)  # (batch, seq_len, embed_dim)
        x = x.permute(1, 0, 2)        # transformer expects (seq_len, batch, embed_dim)
        x = self.transformer(x)
        x = x.mean(dim=0)             # pool sequence
        return F.normalize(self.fc(x), p=2, dim=1)
def multitask_loss(user_emb, item_emb, ctr_labels, temperature=0.05):
    # Contrastive retrieval loss
    logits = torch.matmul(user_emb, item_emb.T) / temperature
    retrieval_labels = torch.arange(user_emb.size(0)).to(user_emb.device)
    retrieval_loss = F.cross_entropy(logits, retrieval_labels)

    # CTR prediction (binary classification)
    ctr_logits = (user_emb * item_emb).sum(dim=1)
    ctr_loss = F.binary_cross_entropy_with_logits(ctr_logits, ctr_labels)

    return retrieval_loss + 0.5 * ctr_loss
class ItemTower(nn.Module):
    def __init__(self, item_dim, kg_dim, embed_dim):
        super().__init__()
        self.fc_item = nn.Linear(item_dim, embed_dim)
        self.fc_kg = nn.Linear(kg_dim, embed_dim)

    def forward(self, item_features, kg_features):
        item_emb = self.fc_item(item_features)
        kg_emb = self.fc_kg(kg_features)
        combined = item_emb + kg_emb
        return F.normalize(combined, p=2, dim=1)
def recall_at_k(user_emb, item_emb, k=10):
    scores = torch.matmul(user_emb, item_emb.T)
    topk = scores.topk(k, dim=1).indices
    correct = (topk == torch.arange(user_emb.size(0)).unsqueeze(1).to(user_emb.device)).any(dim=1)
    return correct.float().mean().item()

def ndcg_at_k(user_emb, item_emb, k=10):
    scores = torch.matmul(user_emb, item_emb.T)
    _, indices = scores.topk(k, dim=1)
    gains = 1.0 / torch.log2(torch.arange(2, k+2).float())
    ndcg = (gains * (indices == torch.arange(user_emb.size(0)).unsqueeze(1).to(user_emb.device)).float()).sum(dim=1)
    return ndcg.mean().item()

import wandb

wandb.init(project="two_tower_retrieval", anonymous='allow')

for epoch in range(10):
    user_features = torch.rand(32, user_dim)
    item_features = torch.rand(32, item_dim)
    ctr_labels = torch.randint(0, 2, (32,)).float()

    user_emb, item_emb = model(user_features, item_features)
    loss = multitask_loss(user_emb, item_emb, ctr_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    wandb.log({"loss": loss.item(), "epoch": epoch})
def diversity_penalty(item_embs, alpha=0.1):
    sim_matrix = torch.matmul(item_embs, item_embs.T)
    redundancy = sim_matrix.mean()
    return alpha * redundancy
loss = multitask_loss(user_emb, item_emb, ctr_labels) + diversity_penalty(item_emb)
recall = recall_at_k(user_emb, item_emb, k=10)
ndcg = ndcg_at_k(user_emb, item_emb, k=10)
wandb.log({"recall@10": recall, "ndcg@10": ndcg})
index = faiss.IndexFlatIP(embed_dim)
index.add(item_embs_np)

# Query
D, I = index.search(user_emb_np, k=10)
print("Top-10 retrieved items:", I[0])
