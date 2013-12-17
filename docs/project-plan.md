##4. Project Plan

###4.1 Planning Process
As a group, we planned the project iteratively and stuck to a feature-driven development rather than a component-driven process. That is, for each step of the process, the group discussed the desired (yet feasible at that point in time) feature requirements during the weekly meetings and divided up the work amongst members. Thus, while there were certain long-term milestone established such as "implementing the parser" the team stuck closely with the short-term goals concerning features. A more detailed description of the process is outlined in the 'Project Timeline' section.

### Include a one-page programming style guide used by the team

###4.3 Project Timeline
| Date   			 | Milestone 		 | 
|:-------------|:------------- | 
| Sep. 15  	 | Project proposal drafted| 
| Oct. 07   		 | Lexer completed  	| 
| Oct. 25 		 | Parser completed 		 | 
| Oct. 31			 | Language reference manual drafted |
| Nov. 05			 | Semantic checking and runtime library completed |
| Dec. 04			 | Code generation completed |
| Dec. 18			 | Final report completed |


###4.4 Team Responsibilities
| Team Member   					| Responsibilities           | 
|:------------- 					|:------------- | 
| Howard Mao (team leader)| Compiler Frontend, Runtime Library, Code Generation 		| 
| Zachary Newman					| Compiler Frontend, Code Generation, Test Suite Creation |
| Sidharth Shankar			  | Compiler Frontend, Code Generation, Semantic Checking   |
| Jonathan Yu 					  | Code Generation, Runtime Library  								      |
| Harry Lee   						| Code Generation, Documentation  												|


###4.5 Development Environment

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

###4.6 Project Log

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
| Oct. 15  	| Integrated Regression Test Suite set up|  
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
