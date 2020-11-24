This is a fork of an open-source implementation of BF-score in Python.

The BF-score metric is in the `metric_bfscore.py` script. `bfscore.py` and `evaluate_single_image.py` apply this metric on the test images in `data`.

For more details on the original implementation, please visit the original repository:

| [github.com/minar09/bfscore_python](https://github.com/minar09/bfscore_python)  |
|:------:|
| original repo |

Below is the original `README.md` file.

---

# bfscore_python
Boundary F1 Score - Python Implementation

This is an open-source python implementation of bfscore (Contour matching score for image segmentation) for multi-class image segmentation, implemented by EMCOM LAB, SEOULTECH.

Reference: [Matlab bfscore](https://www.mathworks.com/help/images/ref/bfscore.html)

### Run
To run the function simply run `python bfscore.py` after setting your image paths and threshold.

### Crosscheck
Score cross-checking with built-in Matlab function is done and presented in `bfscore.pptx`.
