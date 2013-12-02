## Test Plan

The test suite for Vector consisted of simple regression tests for language
features, as well as longer tests to demonstrate target programs in the language.

### Rationale

We omitted unit tests, trusting the OCaml type checker to detect major bugs
(such as missing cases in pattern matching). Edge cases were simple to write in
Vector, so our regression test suite includes the sort of edge cases likely to
be included in a unit test suite.

By running tests frequently and before each check-in, we could catch any
backwards-incompatible changes during development. This system also allows
test-driven development, as the test suite can be run with tests that use
unimplemented language features, simply failing until those features are
implemented.

### Mechanism

#### Components

The components of the "Project Plan" section relevant to the test plan are

* **Vagrant** to manage virtual development environments

* **SCons** as a build tool

* **nvcc** to compile generated CUDA code

* **gpuocelot** to run PTX bytecode on the virtual x86 machines

#### Implementation

We created a `test` folder in the Vector source repository with an `SConscript`
file implementing the build procedure. Within a Vagrant virtual machine, run
`scons test` to

1. Using Vector, compile all test programs (with a `.vec` suffix) to CUDA files

2. Using nvcc, compile the generated CUDA sources to ELF executables linked
   with the gpuocelot library

3. Run the executables and compare the outputs to the expected-result files
   (with a `.out` suffix). Any files with differing output are considered to be
   failing tests.

Add a base filename to the `test_cases` list in the `SConscript` file to add a
new test case. Then, add `.vec` and `.out` for the test case.

### Representative Programs

Waiting until we have better example programs to implement this. Should include
generated source.

### Test Suites Used

Waiting until we have a complete test suite.

### Responsibilities

Zachary was responsible for the initial configuration of the test suites. The
implementers of language features were responsible for their own test suites.
