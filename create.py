import torchaudio
import torch

# tạo dummy audio 2 giây
wav = torch.randn(1, 32000)
torchaudio.save("test.wav", wav, 16000)

