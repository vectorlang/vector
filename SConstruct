import os

env = Environment(ENV=os.environ)
Export('env')

env.SConscript(['compiler/SConscript'])
env.SConscript(['test/SConscript'])
env.SConscript(['bench/SConscript'])
Default('compiler')
