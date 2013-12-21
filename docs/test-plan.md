##6. Test Plan

The test suite for Vector consisted of simple regression tests for language
features, as well as longer tests to demonstrate target programs in the language.

###6.1 Rationale

We omitted unit tests, trusting the OCaml type checker to detect major bugs
(such as missing cases in pattern matching). Edge cases were simple to write in
Vector, so our regression test suite includes the sort of edge cases likely to
be included in a unit test suite.

By running tests frequently and before each check-in, we could catch any
backwards-incompatible changes during development. This system also allows
test-driven development, as the test suite can be run with tests that use
unimplemented language features, simply failing until those features are
implemented.

###6.2 Mechanism

####6.2.1 Components

The components of the "Project Plan" section relevant to the test plan are

* **Vagrant** to manage virtual development environments

* **SCons** as a build tool

* **nvcc** to compile generated CUDA code

* **gpuocelot** to run PTX bytecode on the virtual x86 machines

####6.2.2 Implementation

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

###6.3 Representative Programs

An example of a non-trivial program in vector is calculation of the
mandelbrot set.

    __device__ int mandelbrot(int xi, int yi, int xn, int yn,
        float left, float right, float top, float bottom)
    {
        iter := 0;

        x0 := left + (right - left) / float(xn) * float(xi);
        y0 := bottom + (top - bottom) / float(yn) * float(yi);
        z0 := #(x0, y0);
        z := #(float(0), float(0));

        while (iter < 255 && abs(z) < 2) {
            z = z * z + z0;
            iter++;
        }

        return iter;
    }

    int vec_main()
    {
        img_height := 256;
        img_width := 384;

        int shades[img_height, img_width];

        left := float(-2.0);
        right := float(1.0);
        top := float(1.0);
        bottom := float(-1.0);


        pfor (yi in 0:img_height, xi in 0:img_width) {
            shades[yi, xi] = mandelbrot(xi, yi, img_width, img_height,
                                left, right, top, bottom);
        }

        return 0;
    }

The program launches a pfor thread for each pixel of the image which computes
the number of iterations til convergence for that point on the complex plane.

###6.4 Tests Used

* **arrays.vec** ensures that arrays can both be written to and read from 
* **complex.vec** ensures that our native support for complex numbers works correctly
* **dotprod.vec** performs the dot product of two vectors on the GPU
* **float.vec** ensures that floating point arithmetic works
* **functions.vec** ensures that calling functions works correctly
* **hello.vec** ensures that basic print commands work
* **map.vec** tests the higher-order function `map`
* **reduce.vec** tests the higher-order function `reduce`
* **control_flow.vec** ensures that Vector's main control structures--`for`, 
`while`, and `if` work correctly
* **length.vec** ensures that the native function `length` works on arrays.
* **pfor.vec** ensures that our parallel structure `pfor` works.
* **strings.vec** ensures that string printing operations workk
* **inline.vec** ensures that our `inline` macro for injecting CUDA code works
* **logic.vec** ensures that boolean logic works correctly

###6.5 Benchmarks

We benchmarked the performance of vector code on the CPU and GPU using the
mandelbrot example shown earlier. For the GPU, we used the same code as above.
For the CPU, we used similar code, except the pfor statement was replaced
with a for statement, and the inner `mandelbrot` function was no longer a
device function. By measuring how long it took to complete the computation
for increasing image sizes using the `time` builtin function, we were able to
compare how CPU and GPU code scaled with increasing workloads.

The benchmarks were performed on a desktop computer with a 2.5 GHz AMD Phenom
processor and a NVIDIA GeForce 8400 GS GPU. The results are as follows

#### CPU Results

| Width | Height | Time 1 | Time 2 | Time 3 |
|:------|:-------|:-------|:-------|:-------|
| 640   | 480    | 5.16   | 5.16   | 5.16   |
| 800   | 600    | 8.06   | 8.07   | 8.07   |
| 1024  | 768    | 13.22  | 13.22  | 13.22  |
| 1152  | 864    | 16.72  | 16.73  | 16.74  |
| 1280  | 960    | 20.64  | 20.66  | 20.65  |

#### GPU Results

| Width | Height | Time 1 | Time 2 | Time 3 |
|:------|:-------|:-------|:-------|:-------|
| 640   | 480    | 0.19   | 0.19   | 0.19   |
| 800   | 600    | 0.28   | 0.28   | 0.28   |
| 1024  | 768    | 0.47   | 0.46   | 0.46   |
| 1152  | 864    | 0.53   | 0.53   | 0.53   |
| 1280  | 960    | 0.61   | 0.61   | 0.61   |

![Benchmark Results](docs/benchmark-result-plot.png)

###6.6 Responsibilities

Zachary was responsible for the initial configuration of the test suites. The
implementers of language features were responsible for their own test suites.
Howie wrote the Mandelbrot example and benchmarks.
