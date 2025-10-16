`pip install -r requirements.txt`

<pre>python scripts/pipeline.py \
  --config configs/spy-default.yml \
  --algo both \
  --export-plots \
  --export-frame \
  --run-name spy_run \
  --log-level INFO \
  --thresholds "30,40"</pre>

<pre>python scripts/pipeline.py \
  --config configs/qqq-default.yml \
  --algo both \
  --export-plots \
  --export-frame \
  --run-name qqq_run \
  --log-level INFO \
  --thresholds "30,40,50,60,70,80,90,100"</pre>

<pre>python scripts/pipeline.py \
  --config configs/spy-default.yml \
  --algo both \
  --export-plots \
  --export-frame \
  --run-name spy_run \
  --log-level INFO \
  --use-pca \
  --thresholds "30,40,50,60,70,80,90,100"</pre>

<pre>python scripts/pipeline.py \
  --config configs/qqq-default.yml \
  --algo both \
  --export-plots \
  --export-frame \
  --run-name qqq_run \
  --log-level INFO \
  --use-pca \
  --thresholds "30,40,50,60,70,80,90,100"</pre>
