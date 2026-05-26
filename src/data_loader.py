import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import pandas as pd
from pathlib import Path

def load_data(link: str="muhammadwaqas023/ai-impact-in-future-on-jobs-market-in-2030") -> None:
    # Download latest version
    path = kagglehub.dataset_download(link)

    print("Path to dataset files:", path)

    # Set the path to the file you'd like to load
    file = os.listdir(path)
    full_path = os.path.join(path, file[0])
    df = pd.read_csv(full_path)
    print("First 5 rows:\n", df.head())

    #save the data to data/raw/AI_Impcat_Raw.csv at the project root
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "raw"  
    output_path = output_dir / "AI_Impact_Raw.csv"
    output_dir.mkdir(parents=True, exist_ok=True)  
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    return df
