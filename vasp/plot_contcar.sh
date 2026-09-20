#!/bin/bash

for f in CONTCAR_*; do
  base=${f%.*}   # 去掉后缀，自己按需要改
  echo "Processing $f ..."

  # 侧视图
  timeout 25s xvfb-run -a VESTA -open "$f" -rotate_x -90 -flush -export_img scale=4 "${base}_sideview.png" 

  # 俯视图
  timeout 25s xvfb-run -a VESTA -open "$f" -flush -export_img scale=4 "${base}_topview.png" 

done

