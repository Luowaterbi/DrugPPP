import sys
import time

path = sys.argv[1]
cnt = 0
val_score = {"RMSE": 0, "MAE": 0}
test_score = {"RMSE": 0, "MAE": 0}

with open(path, "r") as reader:
    for line in reader:
        cnt += 1
        line.replace("\n", "")
        [name, loss_type, val, test] = line.split(";")
        val_score[loss_type] += float(val)
        test_score[loss_type] += float(test)

cnt /= 2
with open(path, "a") as writer:
    writer.write("AVG : VAL  RMSE : {} VAL  MAE : {}\n".format(val_score["RMSE"] / cnt, val_score["MAE"] / cnt))
    writer.write("AVG : TEST RMSE : {} TEST MAE : {}\n".format(test_score["RMSE"] / cnt, test_score["MAE"] / cnt))

with open("./runs/final.txt", "a") as writer:
    writer.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    writer.write("\n" + path + "\n")
    writer.write("AVG : VAL  RMSE : {} VAL  MAE : {}\n".format(val_score["RMSE"] / cnt, val_score["MAE"] / cnt))
    writer.write("AVG : TEST RMSE : {} TEST MAE : {}\n".format(test_score["RMSE"] / cnt, test_score["MAE"] / cnt))
