##2. Language tutorial

Much of the Vector language is very similar to C. Therefore, to keep this tutorial short, we focus mainly on the differences between Vector and C. We assume readers have a basic understanding of C.

###2.1 Environment Settings

To compile and run Vector Language programs, you must have a CUDA compiler and a working implementation of the CUDA libraries. We developed and tested Vector using the gpuocelot (code.google.com/p/gpuocelot). This tutorial from here on assumes that users have a working environment.

###2.2 Building the Compiler

Compilation of the Vector compiler requires OCaml (version 4.0.0 or higher,  including ocamllex and ocamlyacc) and the SCons build system, which requires Python2.  Simply type ‘scons’ in the build directory and the compiler binary will be named ‘generator’ and located in the ‘compiler’ subdirectory. 

###2.3 Hello World

The Vector language will be very familiar to users who have prior experience with C-like languages. Without further ado, here is the Hello World program in Vector:
 
	int vec_main()
	{
	    printf("Hello, World!\n");
	    return 0;
	}

vec_main, a special function that takes no arguments, is the entry point of your program and the first code that will be executed. The printf function is identical to the C implementation included in stdio.h. The vec_main function should return 0 for non-exceptional conditions.

To compile the above code, save the file as hello.vec and run the command

	generator < hello.vec > hello.cu

The generator reads the Vector code and outputs CUDA code. (Input is on stdin and output is on stdout). The output of the compiler looks like the following:

	#include <stdio.h>
	#include <stdlib.h>
	#include <stdint.h>
	#include <libvector.hpp>

	 int32_t vec_main(void) {
	printf("Hello, World!\n");
	return 0;
	}

	int main(void) { return vec_main(); }

This should give you a basic idea of how Vector code maps to CUDA code. 

To compile the CUDA code to binary, please consult the CUDA manual for instructions on compiling this CUDA for your specific architecture. Be sure that libvector.hpp is in your include path.

Test the program and make sure it prints “Hello, World!”. Congratulations, you have written your first Vector program. The following sections will provide more information on the building blocks of more complex Vector programs. 

###2.4 Variables and Types

Variables can be considered named areas of memory that hold values that can be read and written. Only primitive types are supported; primitives in Vector (which include the various floating point and integral types, in various sizes) can be declared as follows:

	 type_name variable_name;

Alternatively, assignment and declaration can be performed in a single statement (the assigning declaration) using the ‘:=’ operator. For example,

	 a := 42;

Declares a variable named a, and assigns 42 to it. This is exactly equivalent to

	 int a;
	 a = 42;

Note that with assigning declarations it is not necessary to explicitly declare the type of the variable. This will be inferred from the type of the right hand side. In this case the integral literal is assumed to have type ‘int’. To give it another type, simply cast the right hand side to the desired type. For example, 

	 a := float(42);

will declare a variable named a as a single precision floating point with a value equivalent to 42. For more information on the available types, the type system and type inference, please consult the Language Reference Manual.

###2.5 Arrays

As a language designed primarily for parallel computation, Vector and Vector programs rely heavily on arrays. Arrays in Vector can be multidimensional. For example, to declare a 3 x 4 x 5 array of ints, use 

	 int a[3, 4, 5];

To access an element of the array, a similar syntax is used. For instance, 

	 a[i, j, k]

is equivalent to `a[i][j][k]` in C-like languages. 

###2.6 Operators

Operators in Vector are the same as those in C and similar languages. We support the boolean operators (e.g. comparison), arithmetic operators, and assignment operators. For more information, please consult the Language Reference Manual.

A subtle difference in Vector compared with other C-like languages is that for binary operators, both operands must have the same type, or be cast to the same type – no automatic promotion of variables is supported.

###2.7 Control Flow

The Vector language supports standard conditional statements and looping constructs similar to counterparts in C and Java. The if-statement and while-statement are identical:

	//else is optional
	if ( boolean_condition ) {
		// some code
	}
	else {
		//more code
	}

	//while statement
	while ( boolean_condition ) {
		//some code
	}

The for-statement is a looping construct more similar to that found in Python. It loops over iterators, which can either be ranges or array elements. A single for loop can loop over multiple iterators simultaneously.

Ranges are specified in the format a : b : c, where a is the initial value, b is the upper bound (non-inclusive), and c is the amount by which we increment on each iteration. c can be omitted, or it can be negative. a, b, and c need not be integer literals; they may also be expressions. Also, using multiple iterators separated by a comma will produce code equivalent to two nested loops. Example:

	for (i in 0:5:2, j in 0:4) {
		// some code
	}

is equivalent to

	for (int i = 0; i < 5; i += 2) {
		for (int j = 0; j < 4; j++) {
			// some code
		}
	}

in C++.

We also support iterating over arrays, with the basic syntax (where arr is the name of an array):

	for (x in arr) {
		// some code
	}

Iteration over multidimensional arrays uses the array as if it were single dimensional. That is, for a two dimensional array, we would iterator over every element in the first row, then every element in the second row, etc.

Range iterators and array iterators can be combined in the same for statement, with the same “nested” effect as described above. This covers most typical use cases of the for statement in other languages; if finer-grained control is required, a while loop can be used instead.

###2.8 Functions

Function declarations and definitions in Vector are identical to those in C, and can only happen in the global scope. The format for a declaration-definition is, for example:

	int f(int a, int b) {
		return a + b;
	}

Which declares and defines a function f which adds two integers and returns the resulting integers. To call the function, use:

	 result := f(2,4);

Notice that in the above declaration, the type of result is inferred from the return type of f. As in C, functions can have type void if they perform operations but return no result.

###2.9 Higher Order Functions and pfor

The coolest parts about Vector are the parallel processing components which abstract away some of the details that may have confused users when programming in CUDA. Three of the most heavily used programming patterns for GPU processing, (map, reduce, parallel for) have been implemented in Vector.

####2.9.1 map

Higher-order functions `map` and `reduce` in Vector are called with a `@` character
function at the beginning, but besides designation appear and are called like normal
functions.  `map` takes as an argument a function of a single
argument, and an array.  The function that gets passed to `map` must be designated
with the `__device__` directive, as this function is run on the GPU.  The map
operation then applies the function passed in to every element of the array,
and returns a new array with the results.  Note that the argument type of the function
passed must be equivalent to the type of the elements of the array, but the
return type can be different.

Here's an example:

    __device__ float square(float x) {
     return x * x;
    }

    int[] another_function(int inputs[]) {
      squares := @map(square, inputs);
      return squares;
    }

In this example `another_function` returns an array of the squares of
the elements of the input array.

####2.9.2 reduce

The higher-order function `reduce` takes as arguments a function that takes
two arguments and an array.  Reduce then applies the function to combine the elements
of the array.  It does this by first applying the function to pairs of elements
of the array, obtaining a new array containing half the number of the values of the original,
and then by performing the same operation on this new array, and continuing
until a single value has been obtained.

The return value of the function passed to `reduce` must be of the same type
as the elements of the input array, and the types of the arguments of the function
passed to `reduce` must also be of the same type.  Here's an example:

    __device__ int add(int x, int y) {
      return x + y;
    }

    int another_function(int inputs[]) {
      sum := @reduce(add, inputs);
      return sum;
    }

In this example, `another_function` returns the sum of the input array.

####2.9.3 pfor

Although not exactly a higher order function like map and reduce, pfor is another feature of Vector that abstracts away some of the complexities in GPU programming. It works exactly the same as the regular for statement except that the computation will be performed on the GPU - hence, the syntax remains identitcal to the regular for. Here is a simple example of the pfor:

    void pfor_example()
    {
        int arr[1000, 2];

        pfor (i in 0:len(arr, 0), j in 0:len(arr, 1))
            arr[i, j] = 2 * i + j;
    }
