## Architecture and Design

### Architecture

The main components of the Vector compiler are the scanner, parser, code generator,
and runtime library. We used the Ocaml libraries `ocammlex` and 
`ocamlyacc` to build our scanner and parser respectively.  The code generator
is also built in Ocaml and is implemented in the file `generator.ml`.  The main
entry point to the generator is a function called `generate_toplevel`, which takes
in the abstract syntax tree returned by our parser as an argument and returns
the generated CUDA code as a string, and a record called `environment`, which 
stores important global state about our program.  

`generate_toplevel` steps through the AST of our program, and for each object
in the AST it encounters, we call a corresponding generator function, which
returns the CUDA code for that object in the AST.  After a single
pass through the AST, the compiler has generated the vector code for the input 
program.

However, not all the CUDA code can be generated in the same order that the
corresponding objects appear in the AST.  For example, some of the functions
require forward declarations that need to appear at the beginning of the compiled
CUDA file.  To handle these, we do deferred generation for some of the generated
code.  As we encounter functions for which generation needs to be deferred,
we add necessary information about that function to our `environment` record.

More information about the `environment` record and deferred generation follow
in the next two sections.

#### Environment and Deferred Generation

We need to store global state in our compiler for the following reasons:

* Scope information
* Deferred Generation of functions

Information about scope is essential because the scope that we are in in the AST
has impact on the validity of programs.  For example, we we are declaring a variable
`x`, but a variable `x` has already been declared in the same scope, this is an error,
as variables can only be declared once.

The other reason we need global state is because not all of the code that our
compiler generates is inline--for example, in our generated code we need to begin
a file with the forward declarations of all the functions that will run on the
GPU.  Since we need to add these to the beginning of a file, we simply save
these as global state as we collect more, and then when our compiler has finished
a single pass through the AST of our program outputs the forward declarations before
the generated code from our AST.

In order to keep track of this global state, we defined a record `environment`
in Ocaml with the following fields:

*  `kernel_invocation_functions`
*  `kernel_functions`
*  `func_type_map`
*  `scope_stack`
*  `pfor_kernels`
*  `on_gpu`

In CUDA, the way we run functions on the GPU is through functions called
kernels.  Kernels are special functions, specified with the directive `__device__`
and have special requirements.  Kernels are invoked from code running on the CPU
through specifying the number of blocks and threads we need to run for that function,
and the function name.  A typical kernel invocation looks like this:


    kernel<<<grid_dim, block_dim>>>(*a, *b, size);  // *)

Where `a` and `b` are both pointers to arrays and size is the size of that array.

When we compile a Vector program, when high-level functions `@map` and `@reduce`
and called, we need to invoke a kernel, and in addition, need to compile the
function for the GPU as well.

Therefore, we need to generate two functions for each invocation of a high-level
functions, one for the kernel itself, and one for the invocation of the kernel.
We add information we need to generate the kernel invocatino to the 
`kernel_invocation_list`, and the information the compiler needs to generate
the GPU version of the function to the `kernel_functions` list in the 
environment. We the defer the generations of these functions because high-level
functions can be called in other functions, and nested function definition
is not permitted in CUDA.

To ensure that kernel functions are defined when the kernel invocation functions
are compiled, we add forward declarations to our generated CUDA code for kernel
functions.

`func_type_map` is a map that maps function identifiers to a tuple containing
information about whether that file is meant to be compiled for the GPU, the
return type of the function, and a list containing the types of its arguments.
We use this for type inference, checking if a function call is valid, and for
forward declarations.

`scope_stack` is a stack containing a list of variables that have been declared
in the current scope.  We push and pop from this stack as change scopes as we
walk through the AST, and use this to check if variable declarations are valid.

`pfor_kernels` is a list of kernel functions, each of which corresponds to
a `pfor` loop in the vector code. For each `pfor` loop encountered in our program,
we need to generate a kernel and invoke it.  Since a `pfor` loop could be called
within another function, we need to defer the generation of this kernel, and
therefore add the information that we need to the GPU.

`on_gpu` is a boolean flag that checks to see whether we should be compiling
code for the GPU or for the CPU.  It gets set to true when we start compiling
functions for the GPU, and makes subtle changes, for example using our runtime
library functions for array access in GPUs instead of normal array access.

### Runtime Library
