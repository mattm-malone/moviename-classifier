import csv
import sys

# Removes from IMDB database:
# -Movies with no genre
# -Adult Movies
# -Non Movies

filename = sys.argv[1]

with open(filename, "rt", encoding='ISO-8859-1') as t:
  data = csv.reader(t, delimiter='\t')
  with open("new.tsv", "wt") as f:
      writer = csv.writer(f, delimiter='\t')
      for row in data:
        #print(row[0])
        canWrite = True
        try:
          if row[8] == '\\N':
            canWrite = False
        except:
          pass
        if str(row[1]) != 'movie':
          canWrite = False
        if row[4] == 1:
          canWrite = False
        if canWrite == True:
          print(row[0])
          writer.writerow([row[2], row[8]])