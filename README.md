# Building Damage Assessment Using Satellite Imagery for Large Natural Disasters 

## Project Description

This project focuses on detecting and classifying the extent of building damage caused by disasters using satellite imagery. Leveraging a Siamese U-Net model, the system performs precise damage assessment to support disaster response and recovery efforts. The project includes tools for data preprocessing, model training, and evaluation, as well as comprehensive documentation for ease of use.

## Features
- **Damage Detection and Classification:** Utilizes a Siamese U-Net model to assess building damage levels. (work in progress) 
- **Data Preprocessing:** Generates masks from post-disaster images and categorizes datasets based on disaster types.
- **Modular Codebase:** Well-structured code for easy modifications and improvements.

## Repository Structure
```
├── damage_assessment.py       # Main code for damage detection and classification (work in progress) 
├── mask_polygons.py           # Creates masks from JSON files for model training
├── split_into_disasters.py    # Categorizes dataset based on disaster type
└── Documentation/            # Detailed project documentation
```

## Usage
- **Damage Assessment:**
  ```bash
  python damage_assessment.py
  ```
- **Mask Generation:**
  ```bash
  python mask_polygons.py
  ```
- **Dataset Categorization:**
  ```bash
  python split_into_disasters.py
  ```

Refer to the `Documentation/` folder for detailed information about the project.

