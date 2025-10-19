import numpy as np, polars as pl
from pathlib import Path

p = Path("lake/raw/synthetic.parquet")
p.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.RandomState(0)
k, n, d = 3, 2000, 12
centers = rng.randn(k, d) * 4.0
X = np.vstack([centers[i] + rng.randn(n//k if i<k-1 else n-(n//k)*(k-1), d) for i in range(k)]).astype("float32")
df = pl.DataFrame({f"f{j+1}": X[:, j] for j in range(d)})
df.write_parquet(p.as_posix())
print("Wrote:", p.resolve())
