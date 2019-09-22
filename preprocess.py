
res = []

with open("data_new.csv", "r") as f:
    lines = f.readlines()
    for line in lines:
        if line.split(",")[7] <= 200:
            continue

        else:
            res.append(line)
with open("data_processed.csv", "r") as f:
    f.writelines(res)