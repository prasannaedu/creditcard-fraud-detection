import json
import os
import matplotlib.pyplot as plt
import pandas as pd

def main(outdir="outputs"):
    metrics_file = os.path.join(outdir, "metrics.json")
    if not os.path.exists(metrics_file):
        print(f"[ERROR] {metrics_file} not found. Run training first.")
        return
    
    # Load metrics
    with open(metrics_file, "r") as f:
        results = json.load(f)
    
    # Convert to DataFrame for easy viewing
    df = pd.DataFrame(results).T
    print("\n=== Model Performance ===")
    print(df.round(4))
    
    # Plot metrics
    df.plot(kind="bar", figsize=(10, 6))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "metrics_bar.png"))
    plt.show()

if __name__ == "__main__":
    main()
