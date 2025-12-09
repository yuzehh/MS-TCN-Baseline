# Video Feature Extraction + MS-TCN Evaluation Pipeline

This repository contains two components, forming a complete workflow for extracting video features and training/evaluating an MS-TCN model on our own dataset:

1. **`video_features/`** — Extract per-frame I3D RGB + Flow features  
2. **`ms-tcn/`** — Train and evaluate an MS-TCN model on the extracted features  

You must complete **both steps in sequence**.

---

## 🔧 1. Extract Per-Frame I3D Features

The `video_features/` folder contains code for computing I3D features (Original repo: https://github.com/v-iashin/video_features).  

This step outputs **per-frame 2048-dimensional features** (1024 RGB + 1024 Flow).

```bash
cd video_features

# install environment
conda env create -f conda_env.yml

# load the environment
conda activate video_features

# extract I3D features for the Stanford sample video `1_00_00-00_10.mp4`
python main.py \                                       
    feature_type=i3d \
    device="cuda:0" \
    stack_size=21 \
    step_size=1 \
    video_paths="[./1_00_00-00_10.mp4]" \
    on_extraction="save_numpy" \
    output_path="./output" \
    streams="rgb"
# This follows the same setup used in MS-TCN. For detailed descriptions of each parameter, please see: https://v-iashin.github.io/video_features/models/i3d/
```

The extraction produces **RGB feature files per video** in the `output` folder:

* `1_00_00-00_10_rgb.npy` — (N, 1024) RGB features, N is the frame number

By changing the `streams="flow"`, you could get the **flow feature files per video**. 

You must **concatenate the RGB features and the flow features** to be used as MS-TCN input.

---

# 📂 **2. Prepare Dataset for MS-TCN**

MS-TCN expects the dataset to follow the same structure as the provided example dataset:

```
ms_tcn/data/gtea/
```

Just organize the extracted features into the **same structure**.

---

# 🧠 **3. Train & Evaluate MS-TCN**

We use the code inside `ms-tcn/`, which is a lightly modified version of the original repository: https://github.com/yabufarha/ms-tcn updated for Python 3 compatibility.

```bash
cd ms_tcn

# Example command for training:
python3 main.py --action=train --dataset=gtea --split=1

# Example command for evaluation:
python eval.py --dataset=gtea --split=1
```

The results are in `ms-tcn/results/` folder. 

---

# 📊 **Workflow Summary**

```
RAW VIDEOS
   ↓
(video_features/) Extract I3D RGB + Flow features (1024 + 1024)
   ↓
Concatenate → 2048-dim features per frame
   ↓
Reorganize into MS-TCN dataset format
   ↓
(ms_tcn/) Train & evaluate MS-TCN
```
