# scripts/build_demos.py
from pipeline.demo_builder import build_demos
topics = {
  "hanfu": ("data/raw_zero_shot_logs/hanfu.txt",  "data/demos/hanfu.json"),
  "agri":  ("data/raw_zero_shot_logs/agriculture.txt","data/demos/agri.json"),
  "life":  ("data/raw_zero_shot_logs/local_life.txt","data/demos/life.json"),
}
for t,(src,dst) in topics.items():
    build_demos(src, dst, brand_ctx="", exposure_goals=[], k=4)
print("demos built.")
