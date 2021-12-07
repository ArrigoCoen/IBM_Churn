#%%

from Functions/Project_Functions import *

file_name = "test"
x = 4
var = x
my_piclke_dump(var, file_name)


#%%

import imp

MODULE_PATH = "Functions/Project_Functions.py"
MODULE_NAME = "Project_Functions"

modulevar = imp.load_source(MODULE_NAME, MODULE_PATH)

modulevar.printingstatement()
#%%


#%%