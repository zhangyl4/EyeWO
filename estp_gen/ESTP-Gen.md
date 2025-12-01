## ESTP-Gen

### Overview

ESTP-Gen is the data generation pipeline used in `videollm-online` to construct ESTP-style video question answering data from Ego4D.  
This document describes the **end-to-end pipeline**, including preprocessing, caption generation, one-to-one and contextual QA generation, and known TODOs.

The overall pipeline is:

- **Preprocess**
  - Download Ego4D
  - Filter videos using narrations
  - Downsample videos to 2 FPS
  - (Optional) Split and move data across machines
- **Caption generation**
  - Run scene and action captioning
  - Merge all captions into a unified JSON
- **One-to-one QA generation**
  - Generate QA from multiple sources (action/scene captions, state-change, goal-step, narration)
  - Merge all QA into a unified format
- **One-to-many**
  - Recall all relevant intervals
  - Refine temporal boundaries
  - Run ESTP generation
- **Many-to-many**
  - Generate contextual QA (relevant and irrelevant contexts)

TODO markers are kept where the implementation is still incomplete.

---

### 1. Download Ego4D (Optional)

We provide a subset of **1,000 videos** uniformly sampled by scene type.  
If you need access to the full Ego4D dataset, please apply directly to Ego4D.

We provide several utility scripts to check dataset completeness and integrity:

- `estp_gen/tool_script/detect_dataset_complete.py`: Check whether all required videos exist.
- `estp_gen/tool_script/clear_data.py`: Detect corrupted videos and all-black videos.
- `estp_gen/tool_script/del_notClear_data.py`: Delete problematic videos.

You can modify these scripts according to your own needs.

---

### 2. Filter Data

We use the **base narrations** converted by `videollm-online` as the starting point for filtering.  
Following the design in EgoVLP, we filter out narrations that are either too sparse or too dense.

- Code: `estp_gen/ego4d/ego4d_narration_clear.ipynb`

You can adjust the filtering hyperparameters in the notebook.

---

### 3. 2 FPS Downsample, Split, and Move Data

To make storage and data loading more efficient, we downsample the filtered videos to **2 FPS**  
(current SOTA video-LLMs rarely use a higher sampling rate).

- `estp_gen/tool_script/ffmpeg_highImage.py`: Script to downsample videos.
  - First, put the required video names into a JSON file.
- `estp_gen/tool_script/copy_file.py`: Copy the required subset of data to a specific folder, which is convenient for generation and data transfer.

You can adjust the processing order and paths as needed.

---

### 4. Run Scene and Action Captioning

Make sure to **update all path arguments** in the following scripts before running them.

- `estp_gen/ego4d/video_caption_action.py`  
  Extract action and object nouns from narrations and ask the model to produce detailed descriptions.

- `estp_gen/ego4d/video_caption_scene.py`  
  Use SigLIP to compute frame-wise similarity and segment the video into multiple scenes.

- `estp_gen/ego4d/video_caption_key_segment.py`  
  Only generate captions for segments where:
  - There is likely a large egocentric viewpoint change, and/or  
  - SigLIP visual similarity fluctuates strongly.

---

### 5. Merge Captions

Merge the captions of each video into a unified JSON file:

- `estp_gen/ego4d/video_caption_merge.ipynb`

---

### 6. One-to-One QA Generation

We use the following mapping table to generate **different types of questions** to ensure diversity and temporal quality.  
The **text and timestamp sources** determine the **question type**.

1. **Ego4D action captions** (timestamps localized via actions)  
   - Question types: `OR`, `AP`, `TRU`, `OL`, `OSC`, `EOL`, `EOSC`  
   - Code: `estp_gen/ego4d/actioncaption2objectqa.py`

2. **Ego4D scene captions** (only key segments at scene changes)  
   - Question types: `OR`, `AP`, `TRU`, `OL`, `OSC`, `EOL`, `EOSC`, `OFR`, `IFR`  
   - Code:  
     - `estp_gen/ego4d/moveactioncaption2objectqa.py`  
     - `estp_gen/ego4d/moveactioncaption2functionqa.py`

3. **Ego4D state-of-change annotations**  
   - Question types: `OSC`, `EOSC`  
   - Code:  
     - `estp_gen/soc/o_soc.py` (state change caused by **exo** actions)  
     - `estp_gen/soc/c_soc.py` (state change caused by **ego** actions)

4. **Ego4D goal-step annotations**  
   - Question types: `NAR`, `TU`  
   - Code: `estp_gen/egoplan/egoplan_transform.ipynb`
     - *Egoplan part*: experiments with Egoplan annotations  
     - *Goal-step part*: transformation of Ego4D goal-step labels  
     - Refinement: uses DeepSeek to polish context and perform **temporal consistency checks**.

5. **Ego4D narration**  
   - Question type: `AR`  
   - Code: `estp_gen/ego4d/narration2qa.py`

For repeated batch generation:

- `estp_gen/ego4d/run_moveactioncaption2functionqa.py`

After generating QAs from different sources, use the formatting utilities in:

- `estp_gen/ego4d/caption2qa.ipynb` (formatting section)  
to unify the QA format.

---

### 7. Recall All Relevant Intervals (One-to-Many)

We recall all intervals that are **temporally related** to a given query.  
Note: the current implementation feeds **all captions** into the LLM, which results in a **large token cost**.

- Code: `estp_gen/ego4d/judge_relative_v2.py`

---

### 8. Unify Format and Merge All QA

Use the following notebook to **merge time intervals** and **convert to the final format**:

- `estp_gen/ego4d/refine_time.ipynb`

TODO (merge): finalize and document the full merging procedure.

---

### 9. Contextual QA Generation (Many-to-Many)

We generate contextual QA in three flavors: **object-related**, **irrelevant**, and **task-related**.

- **Object-related contextual QA**
  - `estp_gen/contextual_qa/construct_conqa_from_sigleqa.py`
  - `estp_gen/contextual_qa/construct_conqa_from_sigleqa.ipynb`

- **Irrelevant contextual QA**
  - `estp_gen/contextual_qa/construct_irrelevant_context.ipynb`

- **Task-related contextual QA**
  - TODO
