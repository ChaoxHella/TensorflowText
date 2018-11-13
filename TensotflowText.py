import tensorflow as tf
import numpy as np
import matplotlib.pylot as plt
np.random.seed(5)
x_data=np.linspace(-1,1,100)
y_data=np.squard(x_data)*0.4+np.random.randn(*x_data.shape)
