
<h1 align="center">
  TrackNetV6: A Unified Framework for Lightweight and <br/> Robust
  Fast-Moving Tiny Ball Tracking
</h1>


<p align="center">
  <img src="assets/post.png" width="500" />
</p>



<p align="center">
  <b>📄 Supplementary Material:</b>
  <a href="assets/Supplementary/Supplementary Material.pdf"><b>Download PDF</b></a>
</p>


<h2>TrackNetV6 Framework</h2>
  <p align="center">
    <img src="assets/Figure3.jpg" width="900" />
  </p>



<h2>Visualization</h2>
<div align="center">
  <img src="assets/BA/1_07_06.gif"  width="200" />
  <img src="assets/BA/2_08_12.gif"  width="200" />
  <img src="assets/BA/1_02_00.gif"  width="200" />
  <img src="assets/BA/Thailand.gif" width="200" />
</div>
<div align="center">
  <img src="assets/TA/005.gif" width="200" />
  <img src="assets/TA/018.gif" width="200" />
  <img src="assets/TA/019.gif" width="200" />
  <img src="assets/TA/027.gif" width="200" />
</div>
<div align="center">
  <img src="assets/TE/8-9.gif"       width="200" />
  <img src="assets/TE/9-8.gif"       width="200" />
  <img src="assets/TE/10-11.gif"     width="200" />
  <img src="assets/TE/Wimbledon.gif" width="200" />
</div>
<div align="center">
  <img src="assets/SQ/Coll1.gif"   width="200" />
  <img src="assets/SQ/Coll2.gif"   width="200" />
  <img src="assets/SQ/Coll3.gif"   width="200" />
  <img src="assets/SQ/Coll4.gif"   width="200" />
</div>
<div align="center">
  <img src="assets/Heatmaps/2_18_11.gif"  width="200" />
  <img src="assets/Heatmaps/010.gif"      width="200" />
  <img src="assets/Heatmaps/Clip9.gif"    width="200" />
  <img src="assets/Heatmaps/Clip22.gif"   width="200" />
</div>


<h2>Heatmap Overlays</h2>
<div align="center">
  <img src="assets/overlay/match3-1.gif"  width="200" />
  <img src="assets/overlay/match3-2.gif"  width="200" />
  <img src="assets/overlay/match3-3.gif"  width="200" />
  <img src="assets/overlay/match3-4.gif"  width="200" />
</div>
<div align="center">
  <img src="assets/overlay/match22.gif"   width="200" />
  <img src="assets/overlay/match23.gif"   width="200" />
  <img src="assets/overlay/match24.gif"   width="200" />
  <img src="assets/overlay/match25.gif"   width="200" />
</div>
<div align="center">
  <img src="assets/overlay/stage4-166.png"  width="200" />
  <img src="assets/overlay/stage3-166.png"  width="200" />
  <img src="assets/overlay/stage2-166.png"  width="200" />
  <img src="assets/overlay/stage1-166.png"  width="200" />
</div>
<div align="center">
  <img src="assets/overlay/stage4-167.png"  width="200" />
  <img src="assets/overlay/stage3-167.png"  width="200" />
  <img src="assets/overlay/stage2-167.png"  width="200" />
  <img src="assets/overlay/stage1-167.png"  width="200" />
</div>
<div align="center">
  <img src="assets/overlay/stage4-168.png"  width="200" />
  <img src="assets/overlay/stage3-168.png"  width="200" />
  <img src="assets/overlay/stage2-168.png"  width="200" />
  <img src="assets/overlay/stage1-168.png"  width="200" />
</div>
<div align="center">
  <img src="assets/overlay/stage4-168.png"   width="200" />
  <img src="assets/overlay/stage3-168.png"   width="200" />
  <img src="assets/overlay/stage2-168.png"   width="200" />
  <img src="assets/overlay/stage1-168.png"   width="200" />
</div>



## 📌 Overview
This project provides Python scripts for visualization demos, including:

- ✅ runnable/testable code
- ✅ pretrained weights
- ❗ Note: training and the complete inference pipeline are not released during the anonymous submission stage. The demo code provided here is for reference only and may differ from the implementation in the paper; the full official code will be released after the paper is accepted.


## 🚀 Quick Start
```bash
python demo.py \
  --tracknet_file models/TrackNetBeta.pt \
  --dataset tennis_game_level_split \
  --save_dir ./prediction \
  --device cuda:0 \
  --tolerance 4
```

## 🛠️ Dependencies
Please install following essential dependencies (see requirements.txt):
```bash
pip install -r requirements.txt
```










