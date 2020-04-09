from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
#from pprint import pprint
import numpy as np

class DelayGen(TimeseriesGenerator):

  def __getitem__(self, idx):

    batch_x = []
    batch_y = []

    i = idx * self.sampling_rate + self.length
    while i < len(self.targets)-1 and len(batch_x) < self.batch_size:
        y = self.targets[i]
        if not np.isnan(y):
            batch_x.append(self.data[i-self.length:i])
            batch_y.append(y)
        i += self.sampling_rate

    return (np.array(batch_x), np.array(batch_y))
