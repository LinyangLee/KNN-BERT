import pandas as pd
import csv

data_path = "/remote-home/dmsong/train_data/glue_data/agnews/"
data = {"sentence":[],
        "label": []}
with open(data_path + "dev.csv", "r", encoding="utf-8") as fr:
    raw_data = csv.reader(fr)

    for lines in raw_data:
        # lines = line.split("\t")
        sent_char = lines[-1].strip()
        label = lines[1].strip()
        if not sent_char or sent_char == "":
            continue
        if not label or label == "" or label== "label":
            continue
        data["sentence"].append(sent_char)
        data["label"].append(label)

pd_data = pd.DataFrame(data)

pd_data.to_csv(data_path + "dev.csv", encoding="utf8", sep="\t", index=False)
