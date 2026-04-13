import sys
import math
import numpy as np

def entropy(zeros, ones):
    tot = zeros + ones
    if tot == 0:
        return 0.0
    e = 0.0
    
    for c in (zeros, ones):
        if c != 0:
            p = c / tot
            e -= p * math.log2(p)
    return e

def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    data = np.genfromtxt(input_path, delimiter="\t", dtype=str, skip_header=1)

    if data.size == 0:
        ent = 0.0
        err = 0.0
    else:
        labels = data[:, -1]
        zeros = np.sum(labels == "0")
        ones = len(labels) - zeros 
        ent = entropy(zeros, ones)

        maj = 0 if zeros > ones else 1
        if zeros == ones:
            maj = 1

        incorrect = ones if maj == 0 else zeros
        err = incorrect / (zeros + ones)

    with open(output_path, "w") as out:
        out.write(f"entropy: {ent:.6f}\n")
        out.write(f"error: {err:.6f}\n")

if __name__ == "__main__":
    main()