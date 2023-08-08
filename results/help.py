import gc

# assume we have a large variable that we no longer need
big_var = list(range(0, 1000000))

# delete the variable
del big_var

# run garbage collection manually
gc.collect()

