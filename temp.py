import pandas as pd 
import numpy as np
import os

names = os.listdir('texts/')

for name in names:
    f = open('texts/' + name)
    lines = f.readlines()
    f.close()
    if len(lines) < 10:
        os.system('rm texts/' + name)


