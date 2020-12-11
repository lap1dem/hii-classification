import os

if not os.path.exists('trained-models'):
    os.makedirs('trained-models')

checkpoint_path = 'trained-models/cp.ckpt'