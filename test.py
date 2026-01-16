import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor

model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

wav, sr = torchaudio.load("test.wav")
inputs = fe(wav.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)

print(out.hidden_states[9].shape)
