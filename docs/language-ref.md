# Vector Language Manual

## Introduction

This is the reference manual for Vector, a programming language for the GPU.

## Lexical Conventions

### Comments

The characters `/*` introduce a comment, and the first `*/` ends the comment.
Single-line comments are also supported and denoted with `//` at the beginning
of a line.

### Identifiers
An identifier is a sequence of letters and digits; the first character must be
alphabetic. The underscore `_` counts as alphabetic. Identifiers are
case-sensitive.

### Keywords

The following identifiers are reserved for language keywords

`int`
`char`
`float`
`bool`
`char`
`int8`
`byte`
`uint8`
`int16`
`uint16`
`int`
`int32`
`uint`
`uint32`
`int64`
`uint64`
`double`
`float`
`float32`
`double`
`float64`
`complex`
`complex64`
`complex128`
`string`
`if`
`else`
`while`
`for`
`pfor`
`do`
`return`

### Constants

Vector has the following constants:

#### Integer Constants

An integer constant is a sequence of digits.

#### Character Constants

A character constant is a single character enclosed in single quotes `' '`.
Single quotes must be preceded by a backslash `\`. The `\` character, along with
some non-graphic characters, can be escaped according to the following rules:

 * Backspace `\b`
 * Newline `\n`
 * Carriage Return `\r`
 * Tab `\t`
 * `\` `\\`

Character constants behave like integers.  Characters are stored in two bytes,
with the integer code for the character stored in the lower-order byte and
0 in the higher-order byte.  For characters of length 2, for example if an
escaped character is used, the integer code for the first character is stored
in the lower-order byte and the integer code for the second character is stored
in the higher-order byte.

#### String Constants

String constants consist of a series of characters delimited by quotation marks.

#### Floating Constants

Floating constants consist of an integer part, a decimal point, a fraction part,
an `e` and a signed exponent. If decimal point is not included, then the `e`
and signed exponent must be included, otherwise, they are optional.

#### Array Constants

Array constants consist of a series of elements of a given type.  Commas are
used to separate individual elements and the array itself is delimited by
`[` and `]`.

## Syntax Notation

In this manual, a `typewriter` typeface indicates literal words and characters.
An *italic* typeface indicates a category with special meaning. Lists are
presented either inline or using bullets. If two items are presented on same
line of a bulleted list separated by commas, they are equivalent.
<epsilon> is used to indicate the empty string.
Backus-Naur Form is used to express the grammar of Vector.

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

Complex number types are given by the following literals:

 * `complex`, `complex64`
 * `complex128`

These two complex types are constructed from two `float32` or two `float64`
types, respectively. The real and imaginary parts of the numbers can be accessed
or assigned by appending `.re` or `.im` to the identifier.

    a := #(3.1, 2.1)
    b := a.re // b is 3.1
    a.im = 1.2 // a is now #(3.1, 1.2)

#### String Types

String types are given by the following literal:

 * `string`

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

Array elements can be assigned to using the same syntax, so `a[4] = 3` will
set the value at index 4 of the array to 3.

Arrays can also be multi-dimensional. Indexing into a multi-dimensional
array is achieved by separating the dimensional index numbers by commas.
So `a[1, 2]` will access row 1, column 2 of the two-dimensional array `a`.

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

#### Scalar types

Any scalar type can be converted to another scalar type.  For any conversion
from a narrower to a wider type, for example from a `int16` to `int32`, or from
`float` to `double`, the conversion can be done with no loss of
precision. If a `double` is converted to a `float`, then the `double` will be rounded
and then truncated to be the length of the `float` type that it is converted to.
For conversions between signed types and unsigned types, if the
signed type had a negative number, then `UINT_MAX + 1`, where `UINT_MAX` is the
maximum unsigned integer of the target type, will be added
to it so that the value is a valid unsigned type.

For conversions between any integer and floating-point type, everything up to
16 bits can be cast to `float` with no loss of precision, and everything up to 32
bits can be cast to `double` with no loss of precision. For any floating-point
to integer type conversions, the fraction portion of the floating-point type is 
discarded.

When a `char` object is converted to an int, its sign is propagated through
the upper 8 bits of the resulting int object.

> *explicit-cast* ::= *primitive-type-specifier* `(` *identifier* `)`

## Expressions

### Postfix Expressions

The operators in postfix expressions group left to right.

> *postfix-expression* ::=

> > *primary-expression*

> > | *postfix-expression*`++`

> > | *postfix-expression*`--`

> > | *postfix-expression`.`identifier*

> > | *postfix-expression`[`expression`]`*

### Operators

#### Unary Operators

> *unary-expression* ::=

> > *postfix expression*

> > | `++`*unary expression*

> > | `--`*unary expression*

> > | *unary-operator cast-expression*

> >  | `sizeof` *unary-expression*

> >  | `sizeof`(*type-name*)

unary operators include `-`, `!`. The `-` unary operator returns the negative
of its operand.  If necessary, the operand is promoted to a wider type. For
unsigned quantities, the negative is computed by subtracting the promoted value
from the largest value of the promoted type and adding 1. For the case that the
 value is 0, the value returned is also 0.

#### Cast Operations

A unary expression with a parenthesized type preceding it causes the value
of the expression to be converted to the type in the parentheses.
> *cast-expression* ::=
> >  *unary expression*

> > | *(type-name)* *cast-expression*

#### Multiplicative Operators

Multiplicative operators include `*`, `/`, and `%`, and group to the right. 
There are two operands and both must have arithmetic types.

> *multiplicative-expression* ::=
> > *multiplicative-expression* * *cast-expression*

> > | *multiplicative-expression* `/` *cast-expression*

> > | *multiplicative-expression* `%` *cast-expression*

The `*` operator denotes multiplication, the `/` operation gives the quotient,
and the `%` operator gives the remainder after a division of the two operands.

#### Additive Operators

The additive operators include `+` and `-` group, and they group left-to-right.
If the operands have arithmetic types, then the appropriate arithmetic operation
is performed.


> *additive-expression* ::=

> > *multiplicative-expression*

> > | *additive-expression* `+` *multiplicative-expression*

> > | *additive-expression* `-` *multiplicative-expression*

The `+` operator gives the sum of the two operands, and the `-` operator gives
the difference.
#### Shift Operators

The shift operators include *<<* and *>>*.  These operators group left to right,
and each operator must be of an integral type.

> *shift-expression* ::=

> > *shift-expression* `<<` *additive-expression*

> > *shift-expression* `>>` *additive-expression*

The value of shift expression *E1 << E2* is interpreted as *E1* left-shifted
by *E2* bits, and *E1 >> E2* is interpreted as *E1* right-shifted *E2* bits.

#### Relational Operators

Relational operators group left-to-right.

> *relational-expression* ::=
> > *shift-expression*

> > *relational-expression* `>` *shift-expression*
> > *relational-expression* `<` *shift-expression*
> > *relational-expression* `<=` *shift-expression*
> > *relational-expression* `>=` *shift-expression*

The operator `>` denotes the greater-than operation, `<` denotes less-than,
`>=` denotes greater-than-or-equal, and `<=` denotes less-then-or-equal.

Each of these operators returns 0 if false and 1 if true, and this result
is always of type `int`.

#### Equality Operators

> *equality-expression* :==

> > *relational-expression*

> > | *equality-expression* `==` *relational-expression*

> > | *equality-expression* `!=` *relational-expression*

The equality operator `=` denotes equal-to, and the operator `!=` denotes
not-equal-to.  These both return 1 if true and 0 if false, and this value
is of type `int`.  These operators have lower precedence than relational
operators.

#### Bitwise AND

> *AND-expression* ::=

> > *equality-expression*

> > | AND-expression `&` *equality-expression*

The `&` operator denotes the bitwise-and operation. It requires both operands
to be of integral types, and the result is the bitwise-and function applied
to the operands.

#### Bitwise Exclusive OR

> *exclusive-OR-expression* ::=

> > *AND-expression*

> > *exclusive-OR-expression* `^` *AND-expression*

The `^` operator denotes the bitwise-exclusive-or operation.  It also requires
both operands to have intergral types, and the result is the bitwise-exclusive-or
operation applied to the two operands.

#### Bitwise Inclusive OR

> *inclusive-OR-expression* ::=

> > *exclusive-OR-expression*

> > | *inclusive-OR-expression* `|` *exclusive-OR-expression*

The `|` operator denotes the bitwise-inclusive-or operation.  It also requires
both operands to have intergral types, and the result is the bitwise-inclusive-or
operation applied to the two operands.

#### Logical AND

> *logical-AND-expression* ::=

> > *inclusive-OR-expression*

> > | *logical-AND-expression* `&&` *inclusive-OR-expression*

The `&&` operator returns 1 if both operands are not equal to 0, and 0 if
at least one operand is equal to 0, and the type of this value is `int`.  It
groups expressions left-to-right. Operands must be of arithmetic types.

#### Logical OR

> *logical-OR-expression* ::=
> > *logical-AND-expression*

> > *logical-OR-expression* `||` *logical-AND-expression*

The `||` operator returns 1 if at least one operand is not equal to 0, and 0
otherwise. The type of this return value is `int`.  This operator groups
expressions left-to-right. Operands must be of arithmetic types.


#### Conditional Operator

> *conditional-expression* ::=
> > *logical-OR-expression*

> > | *logical-OR-expression* `?` *expression* `:` *conditional-expression*

The first expression is evaluated and then compared to 0.  If the result is
not equal to 0, then evaluate the expression on the left-hand side of the
colon symbol, otherwise, evaluate the expression on the right-hand side
of the colon. If both the second and third operands have arithmetic values,
then arithmetic conversions are performed such that this expression returns
the same type.

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

> *primitive-declaration* ::= *primitive-type-specifier* *identifier* `;`

> > | *identifier* `:=` *expression* `;`

The first primitive declaration declares a primitive type variable unintialized.
In this case, the value of the identifier before first assignment is unspecified.

The second declaration declares a primitive variable with the given 
identifier with its initial value set to the result of the expression. 
The type of the identifier will be inferenced from the expression. 
If you wish to specify the exact type of the identifier, use an explicit cast.

### Array Declarations

> *array-declaration* ::= *primitive-type-specifier* *identifier* `[];`

> > | *primitive-type-specifier* *identifier* `[` *index-list* `];`

> > | *identifier* `:=` *expression* `;`

> *index-list* ::= *index-list* `,` *expression* | *expression*

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

> > | *declaration*

> > | *compound-statement*

> > | *selection-statement*

> > | *iteration-statement*

> > | *jump-statement*

Statements in Vector are executed in sequence except as described as part of
compound statements, selection statements, iteration statements, and jump
statements.

### Expression Statements

> *expression-statement* ::= *expression* `;`

An expression statement is an expression with its value discarded followed by
a semicolon. The side effects of the expression still occur.

### Declarations

Declarations are also considered statements. The only caveat is that
nested function declarations are not allowed.

### Compound Statements

> *compound-statement* ::= `{` *statement-list* `}`

> *statement-list* ::= *statement-list* *statement*

> > | <epsilon>

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

> *jump-statement* ::= `return` *expression* `;` | `return` `;`

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
