RUN:

```python
python detection_compare.py --annotation_csv rectangles_342.csv --groundtruth_csv groundtruth.csv
```

```python
python ocr_compare.py --annotation_csv polygon_ocr_346.csv --groundtruth_csv groundtruth.csv
```

```python 
python detection_review.py --annotation_csv rectangles_342.csv
```

```python
python classification_groundtruth_estimation_and_review.py --annotation_csv classification
```