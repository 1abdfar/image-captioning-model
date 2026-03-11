import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageCaptionDataset
from model import EncoderCNN, DecoderRNN

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration
    embed_size = 256
    hidden_size = 512
    attention_dim = 256
    batch_size = 32
    learning_rate = 3e-4
    num_epochs = 10
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Set up data (Ensure you have data in the specified paths)
    dataset = ImageCaptionDataset(root_dir="data/images", captions_file="data/captions.json", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    vocab_size = len(dataset.vocab)
    
    # Initialize models
    encoder = EncoderCNN().to(device)
    decoder = DecoderRNN(attention_dim, embed_size, hidden_size, vocab_size).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(list(decoder.parameters()) + list(encoder.resnet.parameters()), lr=learning_rate)
    
    print(f"Starting training on {device}...")
    for epoch in range(num_epochs):
        for idx, (imgs, captions) in enumerate(dataloader):
            imgs, captions = imgs.to(device), captions.to(device)
            lengths = [len(c) for c in captions]
            
            features = encoder(imgs)
            outputs, _ = decoder(features, captions, lengths)
            
            # Calculate loss (excluding SOS token)
            loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")
                
    # Save checkpoints
    torch.save(encoder.state_dict(), "encoder.pth")
    torch.save(decoder.state_dict(), "decoder.pth")

if __name__ == "__main__":
    train()
    