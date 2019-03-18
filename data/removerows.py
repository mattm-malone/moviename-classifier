import csv
import sys
import enchant

# Removes from IMDB database:
# -Non english movies
# -Movies with no genre
# -Adult Movies
# -TV shows
# -Maybe remove certain genres

filename = sys.argv[1]
d = enchant.Dict("en_US")

with open(filename, "rt", encoding='ISO-8859-1') as t:
  data = csv.reader(t, delimiter='\t')
  with open("new.csv", "wt") as f:
      writer = csv.writer(f, delimiter='\t')
      for row in data:
        canWrite = True
        if row[8] == '\\N':
          canWrite = False
        strRow = str(row[2])
        for word in strRow.split():
          if not word:
            pass
          elif d.check(word) is False:
              canWrite = False
              break
        if str(row[1]) != 'movie':
          canWrite = False
        if row[4] == 1:
          canWrite = False
        if canWrite is True:
          #print(row[0])
          #writer.writerow([row[2], row[8]])
          writer.writerow([row[0], row[1]])