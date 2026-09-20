#!/usr/bin/env bash
set -euo pipefail

# 存放所有结构的目录
mkdir -p structure

# 查找并复制 CONTCAR（跳过指定目录）
find . \
  -type d \( -name freq -o -name CDD -o -name wf -o -name DOS -o -name elf -o -name band -o -name COHP -o -name bader -o -name 'charge*' \) -prune -o \
  -type f -name CONTCAR ! -path './CONTCAR' -print0 |
while IFS= read -r -d '' f; do
  # f 形如 ./folder1/calculation1/CONTCAR
  # 取所在目录路径
  rel_dir="${f%/*}"      # ./folder1/calculation1
  rel_dir="${rel_dir#./}" # 去掉开头的 ./ → folder1/calculation1

  # 把 / 换成 _，得到 folder1_calculation1
  if [[ -z "$rel_dir" || "$rel_dir" == "." ]]; then
    suffix=""
  else
    suffix="_${rel_dir//\//_}"
  fi

  # 目标文件名：CONTCAR_folder1_calculation1
  dest_file="structure/CONTCAR${suffix}"

  # 如果刚好重名（极少见，比如同一路径下有多个 CONTCAR），在后面加编号
  if [[ -e "$dest_file" ]]; then
    i=1
    while [[ -e "${dest_file}_$i" ]]; do
      ((i++))
    done
    dest_file="${dest_file}_$i"
  fi

  cp "$f" "$dest_file"
  echo "Copied: $f  ->  $dest_file"
done
