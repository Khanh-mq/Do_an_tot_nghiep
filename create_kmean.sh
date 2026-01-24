#!/bin/bash

# ============== CÀI ĐẶT BIẾN (bạn chỉnh ở đây) ==============
VI_DATA_DIR="./my_data"          # Folder chứa data gốc
CLIPS_DIR="$VI_DATA_DIR/clips"   # Folder chứa .mp3 từ Common Voice
WAV_DIR="$VI_DATA_DIR/wav_16k"    # Folder output wav 16kHz
MANIFEST_DIR="./manifest_vi"
FEATURE_DIR="./features_vi"
KMEANS_DIR="./kmeans_vi"

K=1000                # Số clusters: 1000 cho enhanced, 500/200/100 cho nhanh/test
LAYER=12              # Layer XLS-R: 9-12 thường tốt, 12 recommend multilingual
CHECKPOINT="facebook/wav2vec2-xls-r-300m"  # Hoặc xls-r-1b nếu GPU mạnh

# ============== BƯỚC 1: Resample tất cả mp3 → wav 16kHz ==============
echo "Bước 1: Resample audio về 16kHz..."
mkdir -p $WAV_DIR

for mp3 in $CLIPS_DIR/*.mp3; do
    if [ -f "$mp3" ]; then
        basename_mp3=$(basename "$mp3" .mp3)
        wav_out="$WAV_DIR/${basename_mp3}.wav"
        ffmpeg -i "$mp3" -ar 16000 -ac 1 "$wav_out" -loglevel error
        echo "Resampled: $basename_mp3"
    fi
done

# Nếu data đã là wav, copy thẳng: cp $CLIPS_DIR/*.wav $WAV_DIR/

# ============== BƯỚC 2: Tạo manifest TSV (dùng script fairseq) ==============
echo "Bước 2: Tạo manifest train.tsv..."
mkdir -p $MANIFEST_DIR

python examples/speech_to_speech/preprocessing/prep_s2st_data.py \
    --audio-dir $WAV_DIR \
    --output-root $MANIFEST_DIR \
    --split train  # Chỉ cần train cho unlabeled

MANIFEST="$MANIFEST_DIR/train.tsv"
echo "Manifest tạo xong: $MANIFEST"

# ============== BƯỚC 3: Extract features từ XLS-R ==============
echo "Bước 3: Extract features layer $LAYER..."
mkdir -p $FEATURE_DIR

python fairseq/examples/textless_nlp/gslm/speech2unit/clustering/extract_xlsr_features.py \
    --manifest $MANIFEST \
    --output-dir $FEATURE_DIR \
    --checkpoint $CHECKPOINT \
    --layer $LAYER

echo "Features extract xong ở: $FEATURE_DIR"

# ============== BƯỚC 4: Train k-means ==============
echo "Bước 4: Train k-means với $K clusters..."
python fairseq/examples/textless_nlp/gslm/speech2unit/clustering/train_kmeans.py \
    --feature_dir $FEATURE_DIR \
    --k $K \
    --output_dir $KMEANS_DIR \
    --checkpoint_path $KMEANS_DIR/kmeans_model.pt

echo "HOÀN THÀNH! K-means model lưu tại: $KMEANS_DIR/kmeans_model.pt"
echo "Sau này extract units dùng: --kmeans_checkpoint $KMEANS_DIR/kmeans_model.pt --layer $LAYER"