# 1. Tạo dict.txt từ train.unit (Đếm tần suất các unit)
python -c "
import re
from collections import Counter

input_file = 'data-bin/train.unit'
output_file = 'data-bin/dict.txt'

print(f'⏳ Đang đọc và làm sạch {input_file}...')

with open(input_file, 'r') as f:
    content = f.read()

# 1. Dùng Regex chỉ tìm các con số đứng riêng lẻ
# Nó sẽ bỏ qua các ký tự '|', chữ cái, hoặc số quá dài (như ID)
tokens = re.findall(r'\b\d+\b', content)

# 2. Lọc kỹ hơn: Chỉ lấy các số < 100 (Vì K-means của mình là 100)
# Bước này loại bỏ hoàn toàn các ID dài ngoằng nếu lỡ bị bắt vào
clean_tokens = [t for t in tokens if int(t) < 100]

# 3. Đếm và ghi file
counter = Counter(clean_tokens)
with open(output_file, 'w') as f:
    for unit, count in counter.most_common():
        f.write(f'{unit} {count}\n')

print(f'✅ Đã tạo xong dict.txt sạch sẽ! Tổng số unit tìm thấy: {len(counter)}')
"

# 2. Tạo config.yaml (Khai báo file dict)
echo "vocab_filename: dict.txt
input_channels: 1
use_audio_input: true
" > data-bin/config.yaml

print('✅ Updated config.yaml')