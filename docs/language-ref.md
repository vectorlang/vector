# Vector Language Manual

## Introduction

This is the reference manual for Vector, a programming language for the GPU.

## Syntax Notation

In this manual, a `typewriter` typeface indicates literal words and characters.
An *italic* typeface indicates a category with special meaning. Lists are
presented either inline or using bullets. If two items are presented on same
line of a bulleted list separated by commas, they are equivalent.

## Types

### Primitive types

Vector supports three categories of primitive types: integers, floating
point numbers, and complex numbers.

#### Integer Types

 * `bool`, `char`, `int8`
 * `byte`, `uint8`
 * `int16`
 * `uint16`
 * `int`, `int32`
 * `uint`, `uint32`
 * `int64`
 * `uint64`

The types starting with `u` are unsigned types. The size (in bits) of each
type is self-explanatory.

#### Floating Point Types

 * `float`, `float32`
 * `double`, `float64`

These two types correspond to IEEE single-precision and double-precision
floating point numbers, respectively.

#### Complex Number Types

 * `complex`, `complex64`
 * `complex128`

These two complex types are constructed from two `float32` or two `float64`
types, respectively. The real and imaginary parts of the numbers can be accessed
or assigned by appending `.re` or `.im` to the identifier.

    a := #(3.1, 2.1)
    b := a.re // b is 3.1
    a.im = 1.2 // a is now #(3.1, 1.2)

### Array types

Arrays are composed of multiple instances of primitive types laid out
side-by-side in memory.

Arrays are a very important part of the Vector language, as they are the
only type that can be modified on both the CPU and GPU. Allocation of arrays
on the GPU and transfer of data between CPU and GPU are handled automatically.

You can index into an array using a square-bracket notation. So, for instance
`a[4]` returns the element at index 4 of the array (arrays are zero-indexed).
You can determine the length of an array using the builtin `len` function.

### Function types

Functions take in one or more array- or primitive- typed variables and
optionally return an array- or primitie- typed variable.

## Declarations

### Primitive Type Declarations

A primitive type variable can be declared unintialized using the syntax

    type-specifier variable-name

For instance,

    int x

### Array Declarations

An array type can be declared using the syntax

    primitive-type-specifier variable-name[]

For instance,

    int arr[]

This does not initialize the array or allocate any storage for it.
You can declare an array of a specific size with uninitialized members using

    primitive-type-specifier variable-name[array-size]

For instance,

    int arr[3]

You can also declare an array and initialize its members using

    primitive-type-specifier variable-name[]{member-list}

For instance,

    int arr[]{1, 2, 3, 4}

### Function declarations

Function declarations take the following form

    return-type function-name(parameter-list) { function-body }

The parameter list is a series of primitive or array declarations separated
by commas. Only the non-initializing primitive declarations and non-sizing
array declarations are allowed.
The function body is just a series of statements.

So, for instance, the following is a valid function declaration.

    float scale_and_sum(float scale, float array[]) {
        float sum = 0.0

        for i in 0:len(array) {
            sum += array[i]
        }

        return scale * sum
    }

## Scope

Vector uses block-level scoping. A block, loosely defined, is a section of
code contained by a function, conditional, or looping construct.
Each nested block creates a new scope, and variables declared in the new
scope supersede variables declared in higher scopes.
