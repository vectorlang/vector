# Vector Language Manual

## Introduction

This is the reference manual for Vector, a programming language for the GPU.

## Syntax Notation

In this manual, a `typewriter` typeface indicates literal words and characters.
An *italic* typeface indicates a category with special meaning. Lists are
presented either inline or using bullets. If two items are presented on same
line of a bulleted list separated by commas, they are equivalent. <EOL> is
used to indicate the end of a line; <epsilon> is used to indicate the empty
string. Backus-Naur Form is used to express the grammar of Vector.

## Types

> *type-specifier* ::=
> *primitive-type* | *array-type* | *function-type* | `void`

Each identifier is associated with a type that determines how it is interpreted.
A void type has no value.

### Primitive Types

> *primitive-type* ::= *integer-type* | *floating-point-type* | *complex-type*

Vector supports three categories of primitive types: integers, floating
point numbers, and complex numbers.

#### Integer Types

Integer types are given by the following literals:

 * `bool`, `char`, `int8`
 * `byte`, `uint8`
 * `int16`
 * `uint16`
 * `int`, `int32`
 * `uint`, `uint32`
 * `int64`
 * `uint64`

The types starting with `u` are unsigned types. The number at the end of a
type name indicates the size of that type and equivalent types in bits.

#### Floating Point Types

Floating-point types are given by the following literals:

 * `float`, `float32`
 * `double`, `float64`

These two types correspond to IEEE single-precision and double-precision
floating point numbers, respectively, as defined in [IEEE 754][].

#### Complex Number Types

Complext number types are given by the following literals:

 * `complex`, `complex64`
 * `complex128`

These two complex types are constructed from two `float32` or two `float64`
types, respectively. The real and imaginary parts of the numbers can be accessed
or assigned by appending `.re` or `.im` to the identifier.

    a := #(3.1, 2.1)
    b := a.re // b is 3.1
    a.im = 1.2 // a is now #(3.1, 1.2)

### Array Types

> *array-type* ::= *primitive-type* `[]`

Arrays are composed of multiple instances of primitive types laid out
side-by-side in memory.

Arrays are a very important part of the Vector language, as they are the
only type that can be modified on both the CPU and GPU. Allocation of arrays
on the GPU and transfer of data between CPU and GPU are handled automatically.

Array elements are accessed using square-bracket notation. For instance `a[4]`
returns the element at index 4 of array `a` (arrays are zero-indexed). The
built-in `len` function returns an `int` representing the length of an array.

### Function Types

> *function-type* ::= *primitive-type* `()` | *array-type* `()`

Functions take in zero or more variables of primitive or array types and
optionally return a variable of primitive or array type.

## Objects and LValues

An object is a named region in memory whose value can be read and modified. An
LValue is an expression referring to an object. It has a value, and a
corresponding region in memory where the value is stored.

An expression is an RValue if it only has a value, but not a corresponding
region in memory, i.e., it cannot have a new value assigned to it. In the
following code:

    a := 4
    b := 5
    a = b

a and b are LValues, because they have a value that can be assigned. 4 and 5
are RValues, because they cannot have new values assigned to them.

LValues are named because they can appear on the left side of an assignment (or
also on the right), whereas RValues can appear only on the right side. All
expressions in Vector are either RValues or LValues.

## Conversions

TODO: Sid by 9/29. We only cast explicitly. Preferred syntax is

> *explicit-cast* ::= *primitive-type-specifier* `(` *identifier* `)`

## Expressions

### Operators

TODO: Sid by 10/6

### Function Calls

> *function-call* ::= *identifier* `(` *argument-list* `)` | *identifier* `()`

> *argument-list* ::= *argument-list* `,` *expression* | *expression*

The type of the identifier must be a function. When a function call is
encountered, each of the expressions in its argument list (if it has one) is
evaluated (with side effects); the order of evaluation is unspecified. Then,
control of execution is given to the function specified by the identifier, with
the a copy of the result of each of the expressions available in the scope of
the function block as *parameters*.

The types of the each of the expressions in the argument list must match exactly
the types of the parameters of the function.

All argument passing is done by-value; that is, a copy of each argument is made
before the function has access to it as a parameter. A function may change the
value of its parameters without affecting the value of the arguments in the
calling function.

However, if an array type is passed to a function, the array is not copied, but
merely the reference to the array. Therefore, any modifications the function
makes to the array affect the value of the array in the calling context.

A function may call itself.

The result of evaluating the function call is the value returned by the function
called.

### Assignment

TODO: Jon by 10/6

## Declarations

> *declaration* ::= *primitive-declaration* | *array-declaration* |
> *function-declaration*

A declaration specifies the type of an identifier; it may or may not allocate
memory for the identifier.

### Primitive Type Declarations

> *primitive-declaration* ::= *primitive-type-specifier* *identifier*

> > | *identifier* `:=` *expression*

The first primitive declaration declares a primitive type variable unintialized.
In this case, the value of the identifier before first assignment is unspecified.

The second declaration declares a primitive variable with the given 
identifier with its initial value set to the result of the expression. 
The type of the identifier will be inferenced from the expression. 
If you wish to specify the exact type of the identifier, use an explicit cast.

### Array Declarations

> *array-declaration* ::= *primitive-type-specifier* *identifier* `[]`

> > | *primitive-type-specifier* *identifier* `[` *expression* `]`

> > | *identifier* `:=` *expression*

> *member-list* ::= *member-list* `,` *expression* | *expression*

The first syntax does not initialize the array or allocate any storage for it.

The second syntax declares an array and allocates storage but does not
initialize its members. The expression is evaluated (with side effects) and the
result is the number of members the array will have (and the size of the array
is the size of the primitive type multiplied by the number of members). The type
of the expression must be an unsigned integer.

The third syntax inferences the type of the array from the expression and
is identical to the initializing declaration for primitives.

### Function Declarations

> *function-declaration* ::=
> *type-specifier* *identifier* `(` *parameter-list* `)` *compound-statement*

> > | *type-specifier* *identifier* `()` *compound-statement*

> *parameter-list* ::= *parameter-list*, *declaration* | *declaration*

A function declaration declares a function that accepts the parameters given by
the parameter list and, when called, evaluates the given block (also known as a
*function body*). A function may not be modified after declaration.

The parameter list is a series of primitive or array declarations separated by
commas. Only the non-initializing primitive declarations and non-sizing array
declarations are allowed. The identifiers specified by the parameter list are
available in the function body.

## Statements

> *statement* ::= *expression-statement*

> > | *compound-statement*

> > | *selection-statement*

> > | *iteration-statement*

> > | *jump-statement*

> > | <epsilon>

Statements in Vector are executed in sequence except as described as part of
compound statements, selection statements, iteration statements, and jump
statements.

### Expression Statements

> *expression-statement* ::= *expression*

An expression statement is an expression with its value discarded. The
side effects of the expression still occur.

### Compound Statements

> *compound-statement* ::= `{` *statement-list* `}`

> *statement-list* ::= *statement*

> > | *statement-list* <EOL> *statement*

A compound statement is also called a *block*. When a block is executed, each of
the statements in its statement list are executed in order. Blocks allow the
grouping of multiple statements, especially where only one is expected. In
addition, scoping is based on blocks; see the "Scope" section for more details.

### Selection Statements

> *selection-statement* ::=
> `if` *expression* *compound-statement* *elseifs* *else*

> *elseifs* ::=
> *elseifs* `else if` *expression* *compound-statement* | <epsilon>

> *else* ::= `else` *compound-statement* | <epsilon>

When a selection statement is executed, first the expression associated with the
`if` is evaluated (with all side effects). If the resulting value is nonzero,
the first substatement is executed. and control flow resumes after the selection
statement. Otherwise, for each else-if clause in the selection statement, the
associated expression is evaluated. If its value is nonzero, its substatement is
executed and control flow resumes after the selection statement; otherwise, the
next else-if clause is checked. If there are no more else-if clause and there is
an else clause, its substatement is executed.

### Iteration Statements

> *iteration-statement* ::= `while` *expression* *compound-statement*
>
> > | `do` *compound-statement* `while` *expression* <EOL>
>
> > | `for` *expression*; *expression*; *expression* *compound-statement*
>
> > | `for` *identifier* `in` *expression* *compound-statement*
>
> > | `pfor` *identifier* `in` *expression* *compound-statement*

When a while statement is reached, the expression is evaluated (with all side
effects). If its value is nonzero, its block is executed, and after the
execution of the block, the while statement is executed again. If the value of
the expression is zero, the execution of the while statement is finished.

A do-while statement behaves identically to the while statement, except that its
block is executed before the expression is checked. This means that the block
executes unconditionally at least once.

A for statement using the first syntax has three expressions. When the for
statement is executed, the first expression (also known as the initialization)
is evaluated with all side effects, and the result is discarded. Then, the
second expression (also known as the condition) is checked. If its value is
nonzero, the block of the for statement executes (the statements in the block
may refer to the identifier). Otherwise, the execution of the for statement is
finished.  After each time the block executes, the third expression is evaluated
with all side effects, and the result is discarded.

The second syntax iterates through the block once for each of the elements of the
expression, which is evaluated once (with side effects) and must be an array
type. In the block, the identifier refers to the current element of the array.

A pfor statement is identical to the second syntax of for statements, but the
iterations happen in parallel on the GPU. The first syntax is not allowed for
pfor loops.

### Jump Statements

> *jump-statement* ::= `return` *expression* <EOL> | `return` <EOL>

A return statement returns control of execution to the caller of the current
function. If the statement has an expression, the expression is evaluated (with
side effects) and the result is returned to the caller.

## External Declarations

TODO: Jon by 10/6. Includes function definitions.

## Scope

Vector uses block-level scoping. A block is another name for a compound
statement (see "Compound Statements" section). Most frequently, a block is a
section of code contained by a function, conditional, or looping construct. Each
nested block creates a new scope, and variables declared in the new scope
supersede variables declared in higher scopes.

## Preprocessing

TODO: Decide whether we need this by 10/6

## Grammar

TODO: all by 10/13

[IEEE 754]: http://www.iso.org/iso/iso_catalogue/catalogue_tc/catalogue_detail.htm?csnumber=57469
