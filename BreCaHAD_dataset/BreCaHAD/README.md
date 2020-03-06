# BreCaHAD
BreCaHAD: A Dataset for Breast Cancer Histopathological Annotation and Diagnosis

Histopathological tissue analysis by a pathologist determines the diagnosis and prognosis of most tumors, such as breast cancer. To estimate the aggressiveness of cancer, a pathologist evaluates the microscopic appearance of the biopsied tissue sample based on morphological features which have been correlated with patient outcome. The **Bre**ast **Ca**ncer **H**istopathological **A**nnotation and **D**iagnosis dataset (BreCaHAD) allows researchers to optimize and evaluate the usefulness of their proposed methods. The task associated with this dataset is to automatically classify the histological structures in these H&E stained images into six classes which are mitosis, apoptosis, tumor nuclei, non-tumor nuclei, tubule, and non-tubule. By providing this dataset to the biomedical imaging community, we hope to encourage researchers in computer vision, machine learning and medical fields to contribute and develop methods/tools for automatic detection and diagnosis of cancerous regions in breast cancer histology images.

![annotation](groundTruth_display/SF-96-2086-A2-0011.png "Annotation Example")

The ground truth is provided in JSON format. Six classes (mitosis,apoptosis, tumor, non_tumor, tubule, non_tubule) are defined. In the given example, there are two mitosis and one tumor annotations. *x* and *y* are the coordinates of the centroid of annotated objects, and the values are between [0, 1].

```json
{
  "mitosis": [
    { "x": 0.502, "y": 0.297 },
    { "x": 0.124, "y": 0.572 }
  ],
   "apoptosis": [],
   "tumor": [
    { "x": 0.320, "y": 0.140 }
   ],
   "non_tumor": [],
   "tubule": [],
   "non_tubule": []
}
```

If you use this code, please cite the following [paper]():

```
@InProceedings{,
   author = {Alper Aksac and Tansel Ozyer and Douglas J. Demetrick and Reda Alhajj},
   title = {BreCaHAD: A Dataset for Breast Cancer Histopathological Annotation and Diagnosis},
   year = 2018
}
```