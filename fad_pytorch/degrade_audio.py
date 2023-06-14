import os

# load files in "real"
real_path = 'real/'
real_files = os.listdir(real_path)

# degrade files and save in "fake"
fake_path = 'fake/'
for file in real_files:
    name_in = real_path+file
    name_out = fake_path+file
    os.system("audio-degradation-toolbox -d degradations.json {name_in} {name_out}".format(name_in=name_in, name_out=name_out))
