```markdown
# AI-Enhanced Spatial Interpolation Workflow

This repository contains a complete workflow for geostatistical spatial interpolation enhanced with AI regression models. The script integrates traditional geostatistical techniques (e.g., variogram analysis) with advanced feature engineering and machine learning (ML) methods. Visualizations use the jet colormap to clearly represent spatial patterns, and all outputs—such as plots, CSV files, and saved models—are automatically organized into output folders.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Workflow Details](#workflow-details)
- [Output Examples](#output-examples)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

This repository implements an advanced geostatistical spatial interpolation workflow that combines traditional variogram-based analysis with AI regression models. The workflow:
- Cleans and visualizes geospatial data over Colombia.
- Computes experimental variograms and fits several theoretical models.
- Engineers multiple feature sets (e.g., full, buffer, nearest neighbors, vector, and coordinates) for AI models.
- Performs 5-fold cross-validation on various AI regression models:
  - Random Forest (with different feature sets)
  - Gradient Boosting Regressor
  - Support Vector Regression (SVR)
  - K-Nearest Neighbors (KNN)
  - Multi-Layer Perceptron (MLP)
- Saves the optimal AI models.
- Generates a prediction grid, computes grid features, and produces spatial predictions.
- Masks predictions to the Colombia boundary and creates detailed spatial maps and histograms.

---

## Features

- **Data Pre-processing:**  
  - Loads a subset of the input dataset.
  - Removes outliers in the target variable ("Apparent Geothermal Gradient (°C/Km)") using the IQR method.
  - Loads and projects the Colombia boundary shapefile.
  
- **Visualization:**  
  - Plots data locations over the Colombia boundary.
  - Displays the distribution of the target variable with a colored histogram (using the jet colormap).

- **Variogram Analysis:**  
  - Computes the experimental variogram.
  - Fits three theoretical models (spherical, exponential, and gaussian) and selects the best based on the mean squared error (MSE).

- **Feature Engineering:**  
  - Generates multiple feature sets for AI models:
    - **Full:** `[x, y, x², y², xy, dist_centroid, mean_nn]`
    - **Buffer:** `[x, y, dist_centroid, min_nn, nn_value]`
    - **NN:** `[x, y]` plus nearest neighbor distances and values.
    - **Vector:** `[x, y, dist_centroid]`
    - **Coords:** `[x, y]` only.
    
- **AI Regression Models:**  
  - Evaluates several models using 5-fold cross-validation:
    - `RF_full`, `RF_vector`, `RF_nn`, `RF_coordinates`
    - `GBR_full`, `GBR_coordinates`
    - `SVR_buffer`
    - `KNN_full`
    - `MLP_nn`
  - Computes performance metrics (RMSE, MAE, R²) and produces both dual-axis comparison plots and per-model scatter plots.

- **Prediction Grid and Mapping:**  
  - Creates a spatial prediction grid within the Colombia boundary.
  - Computes grid features and applies the best AI model to generate predictions.
  - Masks the prediction grid to the Colombia boundary.
  - Produces final spatial scatter maps and histograms, as well as model-specific spatial interpolation maps.

- **Output Management:**  
  - Saves all results (models, plots, CSV files) in organized output folders.

---

## Repository Structure

```plaintext
.
├── Data/
│   ├── data_pre_norm.csv          # Input dataset (subset for demonstration)
│   └── COL_adm0.shp               # Colombia boundary shapefile
├── AI_Output_01/
│   ├── models/                    # Saved optimal AI model objects (pickle files)
│   ├── plots/                     # Generated plots (PNG format)
│   └── results/                   # CSV files with variogram fitting metrics and CV metrics
├── spatial_interpolation_ai.py    # Main script implementing the workflow
└── README.md                      # This file
```

---

## Requirements

- **Python:** 3.x  
- **Key Python Libraries:**
  - `numpy`
  - `pandas`
  - `geopandas`
  - `matplotlib`
  - `gstools`
  - `pykrige`
  - `scikit-learn`
  - `shapely`
  - `scipy`

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your_username/ai-geostatistical-interpolation.git
   cd ai-geostatistical-interpolation
   ```

2. **Install Dependencies:**

   You can install the required libraries using pip:

   ```bash
   pip install numpy pandas geopandas matplotlib gstools pykrige scikit-learn shapely scipy
   ```

---

## Usage

1. **Prepare Your Data:**
   - Place your input dataset (`data_pre_norm.csv`) and the Colombia boundary shapefile (`COL_adm0.shp`) in the `Data/` folder.

2. **Run the Script:**

   Execute the main script:

   ```bash
   python spatial_interpolation_ai.py
   ```

3. **Review Outputs:**
   - All outputs (models, plots, and CSV files) are saved in the `AI_Output_01/` directory.
   - Check the generated plots in `AI_Output_01/plots/` and the metrics/results in `AI_Output_01/results/`.

---

## Workflow Details

The script follows these major steps:

1. **Setup Output Folders:**  
   Automatically creates directories for models, plots, and results.

2. **Data Loading and Cleaning:**  
   Loads the dataset, removes outliers, and loads the Colombia boundary shapefile with proper projection.

3. **Data Visualization:**  
   - Plots data locations on the Colombia map.
   - Creates a histogram of the target variable using the jet colormap.

4. **Variogram Analysis:**  
   - Computes pairwise distances and semivariance.
   - Bins the experimental variogram and fits spherical, exponential, and gaussian models.
   - Selects the best-fit model based on MSE.

5. **Feature Engineering:**  
   - Computes multiple feature sets from the coordinates and derived variables.
   - Incorporates nearest neighbor information to enhance the model inputs.

6. **5-Fold Cross-Validation of AI Models:**  
   - Evaluates several AI regression models mapped to different feature sets.
   - Records performance metrics (RMSE, MAE, R²) and visualizes model performance via comparison and scatter plots.

7. **Saving Optimal Models:**  
   - Trains and saves the optimal AI models based on cross-validation results.
   - Stores the best overall model separately.

8. **Prediction Grid Creation:**  
   - Generates a prediction grid over the Colombia region.
   - Computes grid-specific features in the same manner as for the training data.

9. **Spatial Interpolation Using the Best AI Model:**  
   - Applies the best AI model to the prediction grid.
   - Reshapes and masks the predictions to the Colombia boundary.

10. **Final Visualization:**  
    - Produces a final scatter map and histogram of the best model's predictions.
    - Generates spatial maps and histograms for each AI model with a consistent color scale.

11. **Output Metrics:**  
    - Prints extended CV metrics and identifies the best AI spatial interpolation model based on RMSE.

---

## Output Examples

- **Data Location Map:**  
  Displays the locations of the input data over the Colombia boundary.

- **Target Distribution Histogram:**  
  Shows the distribution of "Apparent Geothermal Gradient (°C/Km)" with color mapping.

- **Variogram Plot:**  
  Plots the experimental variogram along with fitted theoretical models.

- **CV Metrics Comparison:**  
  Dual-axis plots comparing RMSE, MAE, and R² across different AI models.

- **Spatial Interpolation Maps:**  
  Final spatial scatter maps and histograms for the overall best AI model, as well as for each model.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgements

This project utilizes several open-source libraries for geospatial analysis and machine learning. Special thanks to the developers and maintainers of:
- [gstools](https://github.com/GeoStat-Framework/gstools)
- [pykrige](https://github.com/GeoStat-Framework/pykrige)
- [geopandas](https://geopandas.org/)
- [scikit-learn](https://scikit-learn.org/)

Their work makes it possible to integrate advanced spatial interpolation and AI techniques seamlessly.

---
```

Happy Interpolating!
```
