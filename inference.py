import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from model import EncoderCNN, DecoderRNN
# Note: Ensure Vocabulary is imported or properly saved/loaded in a real environment
from dataset import Vocabulary 

def generate_caption(image_path, encoder, decoder, vocab, device, max_len=20):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        features = encoder(img_tensor)
        features = features.view(1, -1, features.size(-1))
        
        h, c = decoder.init_h(features.mean(dim=1)), decoder.init_c(features.mean(dim=1))
        word_idx = torch.tensor(vocab.stoi["<SOS>"]).to(device)
        
        caption = []
        alphas = []
        
        for _ in range(max_len):
            embeddings = decoder.embedding(word_idx).unsqueeze(0)
            attention_weighted_encoding, alpha = decoder.attention(features, h)
            alphas.append(alpha.cpu().numpy())
            
            h, c = decoder.decode_step(torch.cat([embeddings, attention_weighted_encoding], dim=1), (h, c))
            preds = decoder.fc(h)
            
            predicted_word_idx = preds.argmax(dim=1).item()
            predicted_word = vocab.itos[predicted_word_idx]
            caption.append(predicted_word)
            
            if predicted_word == "<EOS>":
                break
                
            word_idx = torch.tensor(predicted_word_idx).to(device)
            
    return " ".join(caption), alphas

if __name__ == "__main__":
    # Placeholder for execution logic. 
    # Example: Load models, load vocabulary, run generate_caption(), plot alphas
    pass