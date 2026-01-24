import sys
import os
import torch
import soundfile as sf
import json
import ast

# Import trực tiếp Model (bỏ qua lớp Vocoder wrapper gây lỗi)
from fairseq.models.text_to_speech.codehifigan import CodeHiFiGANModel

def load_units(unit_file):
    """Đọc file unit"""
    units_dict = {}
    print(f"Reading units from {unit_file}...")
    with open(unit_file, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                sent_id = parts[0]
                try:
                    unit_seq = [int(u) for u in parts[1].strip().split()]
                    units_dict[sent_id] = unit_seq
                except ValueError:
                    continue
    return units_dict

def main():
    # --- CẤU HÌNH ĐƯỜNG DẪN ---
    input_file = "test_unit.txt"
    vocoder_ckpt = "vocoder/g_00500000"
    vocoder_conf = "vocoder/config.json"
    output_dir = "results_audio"
    # --------------------------

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. LOAD CONFIG
    print(f"Loading config from {vocoder_conf}...")
    with open(vocoder_conf) as f:
        cfg = json.load(f)
    
    # [FIX]: Xử lý nếu config bị lồng trong key "model"
    if "model" in cfg:
        print(" -> Flattening nested config...")
        cfg = cfg["model"]

    # 2. KHỞI TẠO MODEL TRỰC TIẾP
    print("Initializing CodeHiFiGANModel...")
    # Chúng ta gọi trực tiếp Model, đảm bảo 'cfg' là dict 100%
    model = CodeHiFiGANModel(cfg)
    model.to(device)
    model.eval()

    # 3. LOAD TRỌNG SỐ (WEIGHTS)
    print(f"Loading weights from {vocoder_ckpt}...")
    state = torch.load(vocoder_ckpt, map_location=device)
    
    # Fairseq checkpoint thường lưu weight trong key 'model' hoặc 'generator'
    if "model" in state:
        state = state["model"]
    elif "generator" in state:
        state = state["generator"]
        
    # Load state dict vào model
    try:
        model.load_state_dict(state)
        print(" -> Model weights loaded successfully!")
    except RuntimeError as e:
        print(f" -> Warning: Key mismatch (thường không sao nếu chỉ thiếu vài key): {e}")
        # Thử load lỏng lẻo hơn nếu cần
        model.load_state_dict(state, strict=False)

    # 4. CHẠY INFERENCE
    data = load_units(input_file)
    if not data:
        print("Error: No units found in input file.")
        return

    print(f"Synthesizing {len(data)} sentences...")
    for sent_id, unit_seq in data.items():
        print(f" -> Generating {sent_id}...")
        
        x = {
            "code": torch.LongTensor(unit_seq).view(1, -1).to(device)
        }
        
        with torch.no_grad():
            try:
                # Gọi hàm forward của model
                # Lưu ý: CodeHiFiGANModel trả về trực tiếp waveform (không qua wrapper)
                wav = model(x["code"])
                
                # Xử lý kết quả đầu ra
                if isinstance(wav, (list, tuple)):
                    wav = wav[0]
                
                wav = wav.squeeze().cpu().numpy()
                
                # Lưu file
                out_path = os.path.join(output_dir, f"{sent_id}.wav")
                sf.write(out_path, wav, 16000)
            except Exception as e:
                print(f"Error processing {sent_id}: {e}")

    print(f"\nDone! Check output in '{output_dir}/'")

if __name__ == "__main__":
    main()