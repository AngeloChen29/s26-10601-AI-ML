import sys
import numpy as np

'''
Training:
1. Open and read the intrain file
2. Discard everything except the final column
3. Count the # of 1's and 0's in the final column; take the max
4. Call this X

1. Open the outtrain file
2. Output X every time in outtrain file
3. Calculate train error
4. Close both files

===========================================================================================
===========================================================================================

Testing:
1. Open the intest & outtest file
2. Output X for all tests in outtest file
3. Calculate test error
4. Close both files


1. Open metricsout
2. Write

error(train): train-error
error(test): test-error

to metricsout
3. Close metricsout
'''

def trainModel(fname) :
  df = np.genfromtxt(fname, skip_header = 1)
  df = df[:, -1]
  ones = 0
  zeros = 0

  for i in df :
    if i == 1 :
      ones += 1
    else :
      zeros += 1

  if ones >= zeros :
    result = 1
  else :
    result = 0
  return result

def testModel(fname, fout, guess) :
  df = np.genfromtxt(fname, skip_header = 1)
  df = df[:, -1]
  f = open(fout, "w")
  disagree = 0

  for i in df :
    f.write('{}\n'.format(guess))
    if i != guess:
      disagree += 1
  
  error = disagree / len(df)
  f.close()
  return error

if __name__ == '__main__' :
  intrain = sys.argv[1]
  intest = sys.argv[2]
  outtrain = sys.argv[3]
  outtest = sys.argv[4]
  outmetrics = sys.argv[5]

  result = trainModel(intrain)
  
  trainerror = testModel(intrain, outtrain, result)
  testerror = testModel(intest, outtest, result)

  f = open(outmetrics, "w")
  f.write('error(train): %f\nerror(test): %f' % (trainerror, testerror))
  f.close()
