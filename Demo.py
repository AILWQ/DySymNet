import numpy as np
from DySymNet import SymbolicRegression
from DySymNet.scripts.params import Params
from DySymNet.scripts.functions import *

# You can customize some hyperparameters according to READEME
config = Params()

# such as operators 
funcs = [Identity(), Sin(), Cos(), Square(), Plus(), Sub(), Product()]
config.funcs_avail = funcs


# Example 1: Input ground truth expression
SR = SymbolicRegression.SymboliRegression(config=config, func="x_1**3 + x_1**2 + x_1", func_name="Nguyen-1")
eq, R2, error, relative_error = SR.solve_environment()
print('Expression: ', eq)
print('R2: ', R2)
print('error: ', error)
print('relative_error: ', relative_error)
print('log(1 + MSE): ', np.log(1 + error))


# Example 2: Load the data file
params = Params()  # configuration for a specific task
data_path = './data/Nguyen-1.csv'  # data file should be in csv format
SR = SymbolicRegression(config=params, func_name='Nguyen-1', data_path=data_path)  # you can rename the func_name as any other you want.
eq, R2, error, relative_error = SR.solve_environment()  # return results
print('Expression: ', eq)
print('R2: ', R2)
print('error: ', error)
print('relative_error: ', relative_error)
print('log(1 + MSE): ', np.log(1 + error))