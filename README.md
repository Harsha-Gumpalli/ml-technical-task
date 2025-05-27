# ml-technical-task 

# 🌿 Plant‑Stress Detection Challenge

Welcome! This Colab notebook presents the **take‑home assignment** you’ll work on over the next 24 h.
***Note : The expected time to complete this task is 2–3 hours, and we will evaluate the outcome based on that time frame. While we understand that a more refined result could be achieved with more time, please don’t spend the whole day on it***

**Goal (open‑ended)** – build *any* pipeline that highlights **stress areas** on plant leaves for three different species.

You may use:

* Classical image processing (colour thresholds, morphology, etc.)
* Unsupervised / self‑supervised or fully‑supervised deep learning
* Hybrid approaches

> **Important:** You are **not required** to hand‑label the entire dataset.  
> Feel free to annotate a *tiny* subset, rely on clustering, or stay fully rule‑based.  
> We will focus on sound reasoning, code clarity, **how easily your method can generalise to new plant types**, and overall result quality—not sheer model size.



## 1 · Dataset

You will receive a ZIP file named **`Technical_interview_images.zip`** containing ≈ 1 200 RGB images

```
Technical_interview_images/
├── demo_images/                  # a handful of small illustration files
│   ├── basil_demo_01.jpg
│   ├── strawberry_demo_01.jpg
│   └── …
├── basil/
│   └── images/
│       ├── basil_0001.jpg
│       ├── basil_0002.jpg
│       └── … 400 files
├── strawberry/
│   └── images/
│       └── … 400 files
└── cucumber/
    └── images/
        └── … 400 files

    
```

A smaller **illustration set** is supplied to show examples of healthy vs. stressed tissue.  

**You can download the zip file containing the images from the following link.**

[Drive Link to Download the dataset 880MB](https://drive.google.com/file/d/1gWiGwoKYU9cufbVq_t_hB-wx2Gn1WRLz/view?usp=drive_link)


### Quick demo — illustration images

#### What does “stress” look like on a leaf?

| Condition           | Typical colour cues                                         | 
| ------------------- | ----------------------------------------------------------- | 
| **Healthy**         | Bright green → pale whitish veins                           | 
| **Early stress**    | Green edge → **yellow-green** centre                        | 
| **Moderate stress** | Edge still green, centre turning **orange / reddish-brown** | 
| **Severe stress**   | Leaf largely **dark-red / brown** or almost black           | 



* A **fully healthy leaf** is uniform green (sometimes with light whitish veins).
* When stress begins, you’ll see a **colour gradient**:

  * **Outside** of the leaf — still green.
  * **Towards the centre** — shifts to yellow-green, then orange/red as stress worsens.
* In severe cases the *entire* leaf can appear dark-red or nearly black.

You’ll build a mask that highlights these stressed (non-green) regions.

**Hover over the images to see a quick description of what it is**

![basil](/demo_images/basil.png "basil")
![cucumber](/demo_images/cucumber.png "cucumber")
![strawberry](/demo_images/strawberry.png "strawberry")
![overall healthy basil](/demo_images/overall%20healthy%20basil.png "overall healthy basil")
![overall healthy cucumber](/demo_images/overall%20healthy%20cucumber.png "overall healthy cucumber")
![overall healthy strawberry](/demo_images/overall%20healthy%20strawberry.png "overall healthy strawberry")
![overall very stressed strawberry](/demo_images/overall%20very%20stressed%20strawberry.png "overall very stressed strawberry")
![overall very stressed cucumber](/demo_images/overall%20very_stressed%20cucumber.png "overall very stressed cucumber")




### ***The following cell shows some examples of stress in the leafs, the images are annotated, put the "demo_images" folder in the workspace before executing the following cell.***
#### **Note : change "/content" with the path to the "/demo_images" folder**



```python
from pathlib import Path
import cv2, matplotlib.pyplot as plt

# 1) collect PNGs -----------------------------------------------------
img_paths = sorted(Path("/content").glob("*.png"))
if not img_paths:
    raise FileNotFoundError("No *.png files found in sample_data/")

# 2) create a tall figure --------------------------------------------
plt.figure(figsize=(6, 4 * len(img_paths)))

for idx, (path, cap) in enumerate(zip(img_paths), start=1):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    plt.subplot(len(img_paths), 1, idx)
    plt.imshow(img)
    plt.title(cap, fontsize=11)
    plt.axis("off")

plt.tight_layout()
plt.show()

```



## 2 · Your tasks

1. **Exploratory analysis** – inspect the images, understand what is stress,and what are the stress patterns.  
2. **Design a pipeline** that, given **one RGB image**, returns a **binary mask** where pixels = 1 if *stressed*.
**Note : Semantic segmentation is not required so we will accept any kind of stress highlighting (Drawing a bounding box for example).**
3. **Prepare a short report / slides** (≤ 5) to justify:
   * your approach & hyper‑parameters
   * one failure example & how you would improve it
   * **Bonus** : How to scale the solution to other plant types

**Time budget:** aim for solutions that run in < 2 min/image during inference on a Colab T4 or a modest CPU laptop (*If the solution needs more computational resources, their use has to be justified*.



## 3 · Submission checklist

* Complete the task in your private fork, then send us the repo link or invite us as a collaborator to your fork to review your solution :
  * `notebook.ipynb` (this file, updated with your work) **or** `solution.py`.
  * 5‑slide PDF.
* Make sure the notebook runs end‑to‑end on a fresh Colab
  * include a requirments.txt file
  * A quick README.md file on how to test the solution is appreciated.

Good luck! 🚀

