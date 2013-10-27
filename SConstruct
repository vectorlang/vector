import os

env = Environment(ENV=os.environ)

env.SConscript(['compiler/SConscript'])
env.SConscript(['test/SConscript'])
Default('compiler')
