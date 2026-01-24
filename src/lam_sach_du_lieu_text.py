import html
import re
import os

input_path = "data_text/train.vi"
output_path = "data_text/clean/train.vi"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        line = html.unescape(line)              # &apos; → '
        line = re.sub(r"\s+'", "'", line)       # xóa space trước '
        line = re.sub(r"'\s+", "'", line)       # xóa space sau '
        fout.write(line)

print("Done")
