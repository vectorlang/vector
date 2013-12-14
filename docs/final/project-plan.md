## Project Plan

### Identify process used for planning, specification, development and testing

### Include a one-page programming style guide used by the team

### Show your project timeline

### Identify roles and responsibilities of each team member

### Describe the software development environment used (tools and languages)

Because Vector targets the GPU, a uniform development environment is critical.

We use:

* [Git] for version control and host a shared repository on [GitHub]. This
includes our source code, test suite, virtual environment specification, and
documentation.

* [Vagrant][] to mangage virtual development environments. We specify in a
`Vagrantfile` that we will develop on an Ubuntu 12.04 virtual machine, and
provision using a shell script to install dependencies.

* [SCons][] as a build tool. Its Python syntax allows more flexibility than
make. We build our compiler (including the parser and scanner) and run our test
suite with SCons.

* [OCaml][] (as well as `ocamllex` and `ocamlyacc`) to write our compiler.

* [nvcc][] to compile generated CUDA code.

* [gpuocelot] to run PTX bytecode on the virtual x86 machines and other hardware
  without NVidia GPUs.

The developer only has to deal with installing Git and Vagrant; our build system
abstracts the details of nvcc and gpuocelot away.

### Include your project log

[Git]: http://git-scm.com/
[GitHub]: https://github.com/
[OCaml]: http://ocaml.org/
[SCons]: http://www.scons.org/
[Vagrant]: http://www.vagrantup.com/
[nvcc]: http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
[gpuocelot]: https://code.google.com/p/gpuocelot/