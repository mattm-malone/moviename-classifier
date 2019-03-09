import csv
import sys
import enchant
# simple script to delete (kinda) delete non english rows
# and also delete movies with no genre
#python 2.7
filename = sys.argv[1]
d = enchant.Dict("en_US")

with open(filename, "rt", encoding='utf-8') as f:
    data = list(csv.reader(f))

with open("new.csv", "wt") as f:
    writer = csv.writer(f)
    for row in data:
      canWrite = True
      strRow = str(row[0])
      for word in strRow.split():
        if not word:
          pass
        elif d.check(word) is False:
            canWrite = False
      if canWrite is True:
        writer.writerow(row)