import os
import json
import pandas as pd
import  warnings 

def msvtt_json2csv(json_path: str, csv_name = None):
    assert json_path.endswith(".json")
    assert os.path.exists(json_path)
     
    with open(json_path, "r") as f:
        data = json.load(f)
        
    # info = data["info"]
    sentences = data["sentences"]
    sentence_df = pd.DataFrame(sentences)       #(caption,  video_id,  sen_id)
    sentence_df = sentence_df[["video_id", "caption"]]
    
    out_dir = os.path.dirname(json_path)
    if csv_name is None:
        csv_name = os.path.basename(json_path).rsplit(".json", 1)[0]
    
    out_path = os.path.join(out_dir, f"{csv_name}.csv")
    if os.path.exists(out_dir):
        warnings.warn("Old file will be overwritten")
        
    sentence_df.to_csv(out_path, index=False)
    print(f"Saved {len(sentence_df)} rows to {out_path}")

    return
    
if __name__ == "__main__":
    msvtt_json2csv("./msrvtt_data/MSRVTT_data.json")