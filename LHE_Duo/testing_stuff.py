# import torch


# model_strings = [
#     "Agents/NFSP_Model/id=1900_steps=500000_target.model",
#     "Agents/NFSP_Model/id=1900_steps=500000_avg.model",
#     "Agents/NFSP_Model/id=1901_steps=500000_target.model",
#     "Agents/NFSP_Model/id=1901_steps=500000_avg.model"
# ]

# models = []
# for name in model_strings:
#     models.append(torch.load(name))

# for i, model in enumerate(models):
#     print("MODEL:", i)
#     for param in model.parameters():
#         if param.data.isnan().any():
#             print("NaNs found in MODEL", i)
#             print(param)

# import time

# xd = []
# for i in range(3_000_000):
#     xd.append(0)


# before = time.time()
# for i in range(100):
#     xd.append(0)
#     del xd[0]
# print(time.time() - before)


## TEST TIMES FOR DATA STRUCTURES ##
# from collections import deque
# import time
# import random

# queue = deque(maxlen=3_000_000)
# buffer = []
# queue.append(1)
# queue.append(1)
# print(len(queue))

# before = time.time()
# for i in range(3_000_000):
#     queue.append(1)
# print("Time for DEQUE FILL:", time.time() - before)

# before = time.time()
# for i in range(600_000):
#     buffer.append(1)
# print("Time for NORMA FILL:", time.time() - before)



# before = time.time()
# for i in range(100000):
#     queue.append(1)
# print("Time for DEQUE:", time.time() - before)





# before = time.time()
# for i in range(1000):
#     buffer.append(1)
#     del buffer[0]
# print("Time for NORMA FILL:", time.time() - before)


# before = time.time()
# print(random.sample(queue, 256))
# print("Time for sample on deque:", time.time() - before)

# before = time.time()
# print(random.sample(buffer, 256))
# print("Time for sample on buffer:", time.time() - before)


# print(len(queue))