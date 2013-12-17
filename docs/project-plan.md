##4. Project Plan

###4.1 Planning Process
The group planned the project iteratively and stuck to a feature-driven development rather than a component-driven process. That is, for each step of the process, the group discussed the desired (yet feasible at that point in time) feature requirements during the weekly meetings. Although mostly iterative, the planning process also included some initial planning from the beginning. Once the language reference manual was created, and the full scope of the project was sketched out, the team set major long-term milestones and an initial feature set with flexible deadlines. The plan was assessed and altered depending on progress reports in weekly group meetings as well as meetings with Professor Edwards.

###4.2 Specification Process
As mentioned in the previous section, the initial feature set and specifications were planned as soon as the first draft of the language reference manual was written. While the group prioritized the initial feature set, additional features mentioned during meetings were kept in mind and committed to as time allowed.

###4.3 Development Process
Although the organization of a compiler is such that there is some sequence in implementing components is required due to dependencies, the development process was largely shaped by the feature-driven planning process. That is, since the weekly commitments depended on feature goals that the group agreed upon, each member was not restricted to working on any one part. This resulted in members taking charge of the specific feature assigned, writing not only the compiler code but also tests, if necessary. Each feature was often developed in pairs or in smaller groups, depending on its difficulty and time commitment.

###4.4 Testing Process
As mentioned in the development process description, each member contributed to testing, since the the test for a given feature was written by the member who implemented it. With each new feature development, additional tests were written and previous tests were run to check that the modifications kept all other parts intact.

###4.5 Style Guide

#### Ocaml coding style

 * Indentations should be four spaces
 * Exception to rule above is pattern matchings, which should be two spaces,
   followed by the pipe character, another space, and then the pattern
 * For `match` statements, the first pattern should begin with a pipe
   character. This rule also holds for other statements that use the pipe,
   such as `type`. The rule can be ignored if the statement is all on one line.
 * No trailing whitespace at the end of lines
 * Lines should not be much longer that 80 characters

#### C++ coding style

 * Indentations should be tabs (hard tabs). Tabs are taken to be equivalent
   in width to 8 spaces.
 * The opening curly brace of a compound statement (like `if` or `for`), should
   be at the end of the first line. The closing brace should go on a
   separate line.
 * Braces in function declarations are different. Opening brace should be
   on separate line.
 * Lines should not be much longer than 80 characters

###4.6 Project Timeline
| Date   			 | Milestone 		 | 
|:-------------|:------------- | 
| Sep. 15  	 	 | Project proposal drafted| 
| Oct. 07   	 | Lexer completed  			 | 
| Oct. 25 		 | Parser completed 		 	 | 
| Oct. 31			 | Language reference manual drafted |
| Nov. 05			 | Semantic checking and runtime library completed |
| Dec. 04			 | Code generation completed |
| Dec. 18			 | Final report completed |


###4.7 Team Responsibilities
| Team Member   					| Responsibilities           | 
|:------------- 					|:------------- | 
| Howard Mao (team leader)| Compiler Frontend, Runtime Library, Code Generation 		| 
| Zachary Newman					| Compiler Frontend, Code Generation, Test Suite Creation |
| Sidharth Shankar			  | Compiler Frontend, Code Generation, Semantic Checking   |
| Jonathan Yu 					  | Code Generation, Runtime Library  								      |
| Harry Lee   						| Code Generation, Documentation  												|


###4.8 Development Environment

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

###4.9 Project Log

| Date   		| Milestone 		| 
|:----------|:------------- | 
| Sep. 05 	| First team meeting and preliminary planning| 
| Sep. 08 	| Language defined| 
| Sep. 10   | Development environment defined| 
| Sep. 15  	| Project proposal drafted| 
| Sep. 20  	| Language features defined| 
| Oct. 06   | Operations in grammar implemented  	|
| Oct. 07   | Lexer completed  	|
| Oct. 09   | Expressions in grammar implemented  	|
| Oct. 14 	| Control-flow statements in grammer implemented |
| Oct. 15  	| Regression test suite set up|  
| Oct. 19		| Statements in grammer implemented |
| Oct. 24		| Basic code generator implemented |
| Oct. 25 	| Parser completed 	| 
| Oct. 29 	| Scoping and symbol checking implemented for code generation |
| Oct. 30   | Type inference implemented  	|
| Oct. 31		| Language reference manual drafted |
| Nov. 05		| Semantic checking and runtime library completed |
| Nov. 08 	| Higher order function 'map' implemented for code generation |
| Nov. 10 	| For loop and control flow implemented for code generation |
| Nov. 18 	| Higher order function 'reduce' implemented for code generation |
| Nov. 20 	| Inline macros implemented for code generation |
| Dec. 03 	| pfor statements implemented for code generation |
| Dec. 04		| Code Generation completed |

[Git]: http://git-scm.com/
[GitHub]: https://github.com/
[OCaml]: http://ocaml.org/
[SCons]: http://www.scons.org/
[Vagrant]: http://www.vagrantup.com/
[nvcc]: http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
[gpuocelot]: https://code.google.com/p/gpuocelot/
