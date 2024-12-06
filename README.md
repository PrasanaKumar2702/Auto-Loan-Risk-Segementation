
# Auto Loan Risk Segmentation Model

## Objective
Developed a clustering model to segment borrowers based on credit risk levels. This project enhances risk-based pricing strategies and portfolio profitability.

## Features
- **Preprocessing**: Standardizes numerical features for clustering.
- **Clustering**: Groups borrowers into low, medium, and high-risk categories using K-Means.
- **Validation**: Evaluates clustering quality with silhouette scores.

## Project Structure
```
auto-loan-risk-segmentation/
├── auto_loan_risk_segmentation.py  # Main script for preprocessing and clustering
├── README.md                       # Project documentation
├── data/
│   ├── borrower_data.csv           # Input dataset
│   ├── processed_borrower_data.csv # Preprocessed data
├── results/
│   ├── clustered_data.csv          # Clustered data with risk categories
│   ├── silhouette_score.txt        # Silhouette score for validation
│   ├── cluster_centers.png         # Plot of cluster centers
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/auto-loan-risk-segmentation.git
   cd auto-loan-risk-segmentation
   ```

2. Install dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib
   ```

## Usage
1. Place the raw dataset in the `data/` directory as `borrower_data.csv`.
2. Run the main script:
   ```bash
   python auto_loan_risk_segmentation.py
   ```

## Results
- Preprocessed data saved to `data/processed_borrower_data.csv`.
- Clustered data saved in the `results/` directory as `clustered_data.csv`.
- Silhouette score saved in `results/silhouette_score.txt`.
- Cluster center plot saved as `results/cluster_centers.png`.

## Example Dataset
Sample format for `data/borrower_data.csv`:
| Borrower_ID | Income | Loan_Amount | Credit_Score |
|-------------|--------|-------------|--------------|
| 1           | 45000  | 15000       | 680          |
| 2           | 60000  | 20000       | 720          |
| 3           | 30000  | 10000       | 580          |

## Contact
For questions or feedback, please reach out to [Your Name](mailto:your.email@example.com).
