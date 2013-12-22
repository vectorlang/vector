Install vagrant and run `vagrant up` to spin up a VM containing CUDA and ocelot.
You can then use `vagrant ssh` to ssh into the VM. Change directory to `/vagrant`
and you will find the shared folder linked to the repo. If you want to update
the packages on the VM (or if it doesn't work correctly the first time),
simply run `vagrant provision`.

The build system uses [SCons](http://www.scons.org/). SCons will be installed
on the VM during provisioning. Run `scons` in the root directory of this
project ("/vagrant/" on the VM). Available targets are the default target (which
builds the compiler) and 'test' (which runs tests).

You can find documentation in the `doc` directory. If you have pandoc installed,
you can build the documentation into a pdf by running `scons doc`.
