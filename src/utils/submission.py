"""
Kaggle Submission Generation & Formatting Module for UMUD Challenge.
Generates both:
1. Semicolon-delimited CSV: image_id;pa_deg;fl_mm;mt_mm (3 decimal places)
2. Microsoft Excel (.xlsx) spreadsheet formatted identically.
"""

from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import pandas as pd
import numpy as np


def format_predictions_dataframe(
    df: pd.DataFrame,
    image_id_col: str = "image_id",
    pa_col: str = "pa_deg",
    fl_col: str = "fl_mm",
    mt_col: str = "mt_mm",
    round_decimals: int = 3,
) -> pd.DataFrame:
    """
    Cleans, orders, rounds, and validates predictions DataFrame according to competition rules.
    """
    sub_df = pd.DataFrame()
    sub_df["image_id"] = df[image_id_col].astype(str)
    
    # Ensure numerical values are non-negative and properly rounded
    sub_df["pa_deg"] = np.round(np.clip(df[pa_col].astype(float), 0.0, 90.0), round_decimals)
    sub_df["fl_mm"] = np.round(np.clip(df[fl_col].astype(float), 1.0, 300.0), round_decimals)
    sub_df["mt_mm"] = np.round(np.clip(df[mt_col].astype(float), 0.5, 100.0), round_decimals)
    
    # Check for NaN / null values
    if sub_df.isnull().any().any():
        raise ValueError("Submission DataFrame contains null/NaN values!")
        
    return sub_df


def save_competition_submission(
    df: pd.DataFrame,
    output_csv_path: Union[str, Path] = "D:/UltraMuscleAI/submission.csv",
    output_excel_path: Optional[Union[str, Path]] = "D:/UltraMuscleAI/submission.xlsx",
    delimiter: str = ";",
    round_decimals: int = 3,
) -> Dict[str, Any]:
    """
    Exports the submission dataframe to both semicolon-delimited CSV and Excel (.xlsx).
    """
    clean_df = format_predictions_dataframe(df, round_decimals=round_decimals)
    
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Format with exact decimal precision for CSV
    formatted_csv_df = clean_df.copy()
    formatted_csv_df["pa_deg"] = formatted_csv_df["pa_deg"].apply(lambda v: f"{v:.{round_decimals}f}")
    formatted_csv_df["fl_mm"] = formatted_csv_df["fl_mm"].apply(lambda v: f"{v:.{round_decimals}f}")
    formatted_csv_df["mt_mm"] = formatted_csv_df["mt_mm"].apply(lambda v: f"{v:.{round_decimals}f}")
    
    # Export semicolon-delimited CSV (no index)
    formatted_csv_df.to_csv(output_csv_path, sep=delimiter, index=False)
    print(f"[+] Semicolon-delimited CSV saved to: {output_csv_path}")
    
    # Export to Excel (.xlsx) if requested
    excel_status = False
    if output_excel_path:
        output_excel_path = Path(output_excel_path)
        output_excel_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
                clean_df.to_excel(writer, sheet_name="Predictions", index=False)
            print(f"[+] Excel (.xlsx) spreadsheet saved to: {output_excel_path}")
            excel_status = True
        except Exception as e:
            print(f"[!] Warning: Could not write Excel file via openpyxl: {e}")
            
    return {
        "num_rows": len(clean_df),
        "csv_path": str(output_csv_path),
        "excel_path": str(output_excel_path) if excel_status else None,
        "sample_head": clean_df.head(3).to_dict(orient="records"),
    }
