import sys
import os

sys.path.append('lib')
from logic import *

learn = main_logic()

learn.fit(15)

preds = learn.get_preds()

p = preds[0][0]

learn.save('model')
