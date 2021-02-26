# Copyright 2021, Philipp Wicke, All rights reserved.

# Loading data from the WiLI-2018 - Wikipedia Language Identification database
with open("wili-2018/x_train.txt", "r") as x_train_file:
  x_train_lines = x_train_file.readlines()
  x_train_lines = [line.replace(",", "") for line in x_train_lines]

with open("wili-2018/y_train.txt", "r") as y_train_file:
  y_train_lines = y_train_file.readlines()

with open("data/train_data.csv" , "w") as train_out:
  train_out.write("language,text\n")
  for x_line, y_line in zip(x_train_lines, y_train_lines):
    train_out.write(y_line.strip()+","+x_line.strip()+"\n")