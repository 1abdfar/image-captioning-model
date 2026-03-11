import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import json

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
        
    def __len__(self):
        return len(self.itos)
        
    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            for word in sentence.lower().split():
                frequencies[word] = frequencies.get(word, 0) + 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
                    
    def numericalize(self, text):
        tokenized_text = text.lower().split()
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]

class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load captions dictionary (format: {"img_name.jpg": ["caption 1", "caption 2"]})
        with open(captions_file, 'r') as f:
            self.captions_dict = json.load(f)
            
        self.imgs = list(self.captions_dict.keys())
        self.vocab = Vocabulary(freq_threshold)
        
        all_captions = [cap for img in self.imgs for cap in self.captions_dict[img]]
        self.vocab.build_vocabulary(all_captions)
        
    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, index):
        img_id = self.imgs[index]
        caption = self.captions_dict[img_id][0] 
        
        img_path = os.path.join(self.root_dir, img_id)
        img = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
            
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return img, torch.tensor(numericalized_caption)
    
    