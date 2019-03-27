import csv
import sys
import enchant
import re
d = enchant.Dict("en_US")

# Removes from IMDB database:
# -Movies with no genre
# -Adult Movies
# -Non Movies

filename = sys.argv[1]

with open(filename, "rt", encoding='ISO-8859-1') as t:
  data = csv.reader(t, delimiter='\t',)
  with open("new.tsv", "wt", encoding='ISO-8859-1') as f:
      writer = csv.writer(f, delimiter='\t')
      for row in data:
        canWrite = True
        strRow = str(row[2])
        notEng = 0
        words = re.sub(r'\W+\'', " ", strRow)
        for word in words.split():
          if not word:
            pass
          elif d.check(word) is False:
            #print(words)
            notEng = notEng + 1
        if notEng > 2:
          canWrite = False
        try:
          if row[8] == '\\N':
            canWrite = False
            #print("nogenre")
        except:
          pass
        if str(row[1]) not in ['movie', 'tvMovie']:
          canWrite = False
        if row[4] == 1:
          canWrite = False
        if canWrite == True:
          #print(row[0])
          writer.writerow([row[2], row[8]])