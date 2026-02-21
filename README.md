# Neural Network - Pima Indians Diabetes Classification

## Problem
Predict whether a patient has diabetes based on 8 clinical measurements using a neural network binary classifier.

**Dataset**: Pima Indians Diabetes (kaggle) 
**Samples**: 768 patients  
**Features**: 8 (pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree, age)  
**Target**: Diabetes (0 = No, 1 = Yes)

---

## Approach
Applied the same neural network pipeline from the Heart Disease lab to this new dataset:
1. Loaded data 
2. Cleaned data — replaced biologically impossible zero values with column medians
3. Explored data — visualized feature distributions by class
4. Split 80/20 train/test with stratification
5. Scaled features using StandardScaler
6. Applied class weights to handle 65/35 class imbalance
7. Built a Sequential neural network (16 → 8 → 1)
8. Trained with Adam optimizer and Early Stopping (patience=20)
9. Evaluated with accuracy, precision, recall, F1, and confusion matrix

---

## Model Architecture
```
Input (8 features) → Dense(16, ReLU) → Dense(8, ReLU) → Dense(1, Sigmoid)
Total parameters: 289
```

---

## Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 75.32% |
| Precision | 62.12% |
| Recall    | 75.93% |
| F1-Score  | 68.33% |

### Confusion Matrix
|                  | Predicted No | Predicted Yes |
|------------------|--------------|---------------|
| **Actual No**    | 75 (TN)      | 25 (FP)       |
| **Actual Yes**   | 13 (FN)      | 41 (TP)       |

---

## Analysis

### 1. Overall Performance
The model achieved 75.32% accuracy, exceeding the expected range of 70-75%. More importantly, through data cleaning and class weight optimization, the model achieved a recall of 75.93% — correctly identifying 41 out of 54 actual diabetic patients. Both training and validation curves converged smoothly with no signs of overfitting.

### 2. Data Cleaning Impact
Several features contained biologically impossible zero values — glucose, blood pressure, skin thickness, insulin, and BMI cannot be zero in a living patient. These were replaced with column medians before training. This cleaning step significantly improved the quality of information available to the model.

### 3. Handling Class Imbalance
The dataset has a 65/35 class imbalance — 500 non-diabetic vs 268 diabetic patients. Without correction, the model was biased toward predicting "No Diabetes." Class weights were applied during training to penalize missing diabetic patients more heavily, which improved recall from 59.26% to 75.93% — a gain of 16.67 percentage points.

### 4. False Negatives — Clinical Priority
The model produced only 13 false negatives — down from 22 in the baseline model. In a real medical setting, false negatives are the most dangerous error since undetected diabetes can lead to serious complications. Minimizing false negatives was the primary optimization goal 

### 5. Potential Improvements
- Use k-fold cross validation for more reliable evaluation
- Try feature engineering (e.g., glucose/BMI ratio)
- Experiment with ensemble methods like Random Forest
- Tune the classification threshold below 0.5 to further improve recall
---

## Files
```
neural-network-diabetes/
├── pima_diabetes_nn.ipynb       # Complete notebook
├── README.md                    # This file
└── results/
    ├── training_curves.png      # Accuracy & loss plots
    ├── confusion_matrix.png     # Confusion matrix heatmap
    └── metrics_summary.txt      # All metrics in text form
```