##Appendix A - Compiler Source Code Listing

###A.1 scanner.mll

	{ open Parser;; exception Illegal_identifier }

	let decdigit = ['0'-'9']
	let hexdigit = ['0'-'9' 'a'-'f' 'A'-'Z']
	let letter = ['a'-'z' 'A'-'Z' '_']
	let floating =
	    decdigit+ '.' decdigit* | '.' decdigit+
	  | decdigit+ ('.' decdigit*)? 'e' '-'? decdigit+
	  | '.' decdigit+ 'e' '-'? decdigit+

	rule token = parse
	  | [' ' '\t' '\r' '\n'] { token lexbuf }
	  | ';' { SC }
	  | ':' { COLON }
	  | '.' { DOT }
	  | ',' { COMMA }
	  | '@' { AT }
	  | '(' { LPAREN }  | ')' { RPAREN }
	  | '{' { LCURLY }  | '}' { RCURLY }
	  | '[' { LSQUARE } | ']' { RSQUARE }
	  | '=' { EQUAL } | ":=" { DECL_EQUAL }

	  | '+' { PLUS } | '-' { MINUS } | '*' { TIMES } | '/' { DIVIDE }
	  | '%' { MODULO }
	  | "<<" { LSHIFT } | ">>" { RSHIFT }
	  | '<' { LT } | "<=" { LTE }
	  | '>' { GT } | ">=" { GTE }
	  | "==" { EE } | "!=" { NE }
	  | '&' { BITAND } | '^' { BITXOR } | '|' { BITOR }
	  | "&&" { LOGAND } | "||" { LOGOR }

	  | '!' { LOGNOT } | '~' { BITNOT }
	  | "++" { INC } | "--" { DEC }

	  | "+=" { PLUS_EQUALS } | "-=" { MINUS_EQUALS }
	  | "*=" { TIMES_EQUALS } | "/-" { DIVIDE_EQUALS }
	  | "%=" { MODULO_EQUALS }
	  | "<<=" { LSHIFT_EQUALS } | ">>=" { RSHIFT_EQUALS }
	  | "|=" { BITOR_EQUALS } | "&=" { BITAND_EQUALS } | "^=" { BITXOR_EQUALS }

	  | '#' { HASH }

	  | decdigit+ | "0x" hexdigit+
	      as lit { INT_LITERAL(Int32.of_string lit) }
	  | (decdigit+ | "0x" hexdigit+ as lit) 'L'
	      { INT64_LITERAL(Int64.of_string lit) }
	  | floating as lit { FLOAT_LITERAL(float_of_string lit) }
	  | '"' (('\\' _ | [^ '"'])* as str) '"'
	      { STRING_LITERAL(str) }
	  | '\'' ('\\' _ | [^ '\''] | "\\x" hexdigit hexdigit as lit) '\''
	      { CHAR_LITERAL((Scanf.unescaped(lit)).[0]) }

	  | "bool" | "char" | "byte" | "int" | "uint"
	  | "int8" | "uint8" | "int16" | "uint16"
	  | "int32" | "uint32" | "int64" | "uint64"
	  | "float" | "float32" | "double" | "float64"
	  | "complex" | "complex64" | "complex128"
	  | "void" | "string"
	      as primtype { TYPE(primtype) }

	  | "return" { RETURN }
	  | "if" { IF }
	  | "else" { ELSE }
	  | "for" { FOR }
	  | "while" { WHILE }
	  | "pfor" { PFOR }
	  | "in" { IN }

	  | "__sym" decdigit+ "_" letter* { raise Illegal_identifier }
	  | "__device__" { DEVICE }
	  | letter (letter | decdigit)* as ident { IDENT(ident) }

	  | "/*" { comments lexbuf }
	  | "//" {inline_comments lexbuf}

	  | eof { EOF }

	and comments = parse
	  | "*/"                { token lexbuf}
	  | _                   { comments lexbuf}

	and inline_comments = parse
	  | "\n"  {token lexbuf}
	  | _ {inline_comments lexbuf}

###A.2 ast.ml

	exception Invalid_type of string

	type binop = Add | Sub | Mul | Div | Mod
	    | Lshift | Rshift
	    | Less | LessEq | Greater | GreaterEq | Eq | NotEq
	    | BitAnd | BitXor | BitOr | LogAnd | LogOr

	type unop = Neg | LogNot | BitNot

	type postop = Dec | Inc

	type datatype =
	    Bool | Char | Int8
	  | UInt8
	  | Int16
	  | UInt16
	  | Int32
	  | UInt32
	  | Int64
	  | UInt64
	  | Float32
	  | Float64
	  | Complex64
	  | Complex128
	  | String
	  | Void
	  | ArrayType of datatype

	type ident =
	    Ident of string

	type lvalue =
	  | Variable of ident
	  | ArrayElem of ident * expr list
	  | ComplexAccess of expr * ident 
	and expr =
	    Binop of expr * binop * expr
	  | AssignOp of lvalue * binop * expr
	  | Unop of unop * expr
	  | PostOp of lvalue * postop
	  | Assign of lvalue * expr
	  | IntLit of int32
	  | Int64Lit of int64
	  | FloatLit of float
	  | ComplexLit of expr * expr
	  | StringLit of string
	  | CharLit of char
	  | ArrayLit of expr list
	  | Cast of datatype * expr
	  | FunctionCall of ident * expr list
	  | HigherOrderFunctionCall of ident * ident * expr
	  | Lval of lvalue

	type decl =
	    AssigningDecl of ident * expr
	  | PrimitiveDecl of datatype * ident
	  | ArrayDecl of datatype * ident * expr list

	type range =
	    Range of expr * expr * expr

	type iterator =
	    RangeIterator of ident * range
	  | ArrayIterator of ident * expr

	type statement =
	    CompoundStatement of statement list
	  | Declaration of decl
	  | Expression of expr
	  | EmptyStatement
	  | IfStatement of expr * statement * statement
	  | WhileStatement of expr * statement
	  | ForStatement of iterator list * statement
	  | PforStatement of iterator list * statement
	  | FunctionDecl of bool * datatype * ident * decl list * statement list
	  | ForwardDecl of bool * datatype * ident * decl list
	  | ReturnStatement of expr
	  | VoidReturnStatement

	let type_of_string = function
	    | "bool" -> Bool
	    | "char" -> Char
	    | "int8" -> Int8
	    | "uint8" -> UInt8
	    | "int16" -> Int16
	    | "uint16" -> UInt16
	    | "int" -> Int32
	    | "int32" -> Int32
	    | "uint" -> UInt32
	    | "uint32" -> UInt32
	    | "int64" -> Int64
	    | "uint64" -> UInt64
	    | "double" -> Float64
	    | "float" -> Float32
	    | "float32" -> Float32
	    | "float64" -> Float64
	    | "complex" -> Complex64
	    | "complex64" -> Complex64
	    | "complex128" -> Complex128
	    | "void" -> Void
	    | dtype -> raise (Invalid_type dtype)


###A.3 parser.mly

	%{ open Ast %}

	%token LPAREN RPAREN LCURLY RCURLY LSQUARE RSQUARE
	%token DOT COMMA SC COLON AT
	%token EQUAL DECL_EQUAL
	%token PLUS_EQUALS MINUS_EQUALS TIMES_EQUALS DIVIDE_EQUALS MODULO_EQUALS
	%token LSHIFT_EQUALS RSHIFT_EQUALS BITOR_EQUALS BITAND_EQUALS BITXOR_EQUALS
	%token LSHIFT RSHIFT BITAND BITOR BITXOR LOGAND LOGOR
	%token LT LTE GT GTE EE NE
	%token PLUS MINUS TIMES DIVIDE MODULO
	%token LOGNOT BITNOT DEC INC
	%token IF ELSE WHILE FOR PFOR IN
	%token RETURN
	%token DEVICE
	%token HASH
	%token EOF
	%token <int32> INT_LITERAL
	%token <int64> INT64_LITERAL
	%token <float> FLOAT_LITERAL
	%token <string> IDENT TYPE STRING_LITERAL
	%token <char> CHAR_LITERAL

	%right EQUAL PLUS_EQUALS MINUS_EQUALS TIMES_EQUALS DIVIDE_EQUALS MODULO_EQUALS LSHIFT_EQUALS RSHIFT_EQUALS BITOR_EQUALS BITAND_EQUALS BITXOR_EQUALS
	%right DOT
	%left LOGOR
	%left LOGAND
	%left BITOR
	%left BITXOR
	%left BITAND
	%left EE NE
	%left LT LTE GT GTE
	%left LSHIFT RSHIFT
	%left PLUS MINUS
	%left TIMES DIVIDE MODULO
	%right UMINUS LOGNOT BITNOT
	%nonassoc IFX
	%nonassoc ELSE

	%start top_level
	%type <Ast.statement list> top_level

	%%

	ident:
	    IDENT { Ident($1) }

	datatype:
	    TYPE { type_of_string $1 }
	  | TYPE LSQUARE RSQUARE { ArrayType(type_of_string $1) }

	lvalue:
	  | ident { Variable($1) }
	  | ident LSQUARE expr_list RSQUARE { ArrayElem($1, $3) }
	  | expr DOT ident { ComplexAccess($1, $3) }

	expr:
	  | expr PLUS expr   { Binop($1, Add, $3) }
	  | expr MINUS expr  { Binop($1, Sub, $3) }
	  | expr TIMES expr  { Binop($1, Mul, $3) }
	  | expr DIVIDE expr { Binop($1, Div, $3) }
	  | expr MODULO expr { Binop($1, Mod, $3) }
	  | expr LSHIFT expr { Binop($1, Lshift, $3) }
	  | expr RSHIFT expr { Binop($1, Rshift, $3) }
	  | expr LT expr     { Binop($1, Less, $3) }
	  | expr LTE expr    { Binop($1, LessEq, $3) }
	  | expr GT expr     { Binop($1, Greater, $3) }
	  | expr GTE expr    { Binop($1, GreaterEq, $3) }
	  | expr EE expr     { Binop($1, Eq, $3) }
	  | expr NE expr     { Binop($1, NotEq, $3) }
	  | expr BITAND expr { Binop($1, BitAnd, $3) }
	  | expr BITXOR expr { Binop($1, BitXor, $3) }
	  | expr BITOR expr  { Binop($1, BitOr, $3) }
	  | expr LOGAND expr { Binop($1, LogAnd, $3) }
	  | expr LOGOR expr  { Binop($1, LogOr, $3) }


	  | lvalue PLUS_EQUALS expr   { AssignOp($1, Add, $3) }
	  | lvalue MINUS_EQUALS expr  { AssignOp($1, Sub, $3) }
	  | lvalue TIMES_EQUALS expr  { AssignOp($1, Mul, $3) }
	  | lvalue DIVIDE_EQUALS expr { AssignOp($1, Div, $3) }
	  | lvalue MODULO_EQUALS expr { AssignOp($1, Mod, $3) }
	  | lvalue LSHIFT_EQUALS expr { AssignOp($1, Lshift, $3) }
	  | lvalue RSHIFT_EQUALS expr { AssignOp($1, Rshift, $3) }
	  | lvalue BITOR_EQUALS expr  { AssignOp($1, BitOr, $3) }
	  | lvalue BITAND_EQUALS expr { AssignOp($1, BitAnd, $3) }
	  | lvalue BITXOR_EQUALS expr { AssignOp($1, BitXor, $3) }

	  | MINUS expr %prec UMINUS { Unop(Neg, $2) }
	  | LOGNOT expr { Unop(LogNot, $2) }
	  | BITNOT expr { Unop(BitNot, $2) }

	  | lvalue DEC { PostOp($1, Dec) }
	  | lvalue INC { PostOp($1, Inc) }

	  | LPAREN expr RPAREN { $2 }

	  | lvalue EQUAL expr { Assign($1, $3) }
	  | lvalue            { Lval($1) }

	  | INT_LITERAL                 { IntLit($1) }
	  | INT64_LITERAL               { Int64Lit($1) }
	  | FLOAT_LITERAL               { FloatLit($1) }
	  | HASH LPAREN expr COMMA expr RPAREN { ComplexLit($3, $5) }
	  | STRING_LITERAL              { StringLit($1) }
	  | CHAR_LITERAL                { CharLit($1) }
	  | datatype LPAREN expr RPAREN { Cast($1, $3) }
	  | LCURLY expr_list RCURLY     { ArrayLit($2) }

	  | ident LPAREN RPAREN               { FunctionCall($1, []) }
	  | ident LPAREN expr_list RPAREN { FunctionCall ($1, $3) }
	  | AT ident LPAREN ident COMMA expr RPAREN
	      { HigherOrderFunctionCall($2, $4, $6) }

	expr_list:
	  | expr COMMA expr_list { $1 :: $3 }
	  | expr                 { [$1] }

	decl:
	  | ident DECL_EQUAL expr SC               { AssigningDecl($1, $3) }
	  | datatype ident SC                      { PrimitiveDecl($1, $2) }
	  | datatype ident LSQUARE RSQUARE SC      { ArrayDecl($1, $2, []) }
	  | datatype ident LSQUARE expr_list RSQUARE SC { ArrayDecl($1, $2, $4) }

	statement:
	  | IF LPAREN expr RPAREN statement ELSE statement
	      { IfStatement($3, $5, $7) }
	  | IF LPAREN expr RPAREN statement %prec IFX
	      { IfStatement($3, $5, EmptyStatement) }

	  | WHILE LPAREN expr RPAREN statement { WhileStatement($3, $5) }
	  | FOR LPAREN iterator_list RPAREN statement { ForStatement($3, $5) }
	  | PFOR LPAREN iterator_list RPAREN statement { PforStatement($3, $5) }

	  | LCURLY statement_seq RCURLY { CompoundStatement($2) }

	  | expr SC { Expression($1) }
	  | SC { EmptyStatement }
	  | decl { Declaration($1) }

	  | RETURN expr SC { ReturnStatement($2) }
	  | RETURN SC { VoidReturnStatement }


	iterator_list:
	  | iterator COMMA iterator_list { $1 :: $3 }
	  | iterator { [$1] }

	iterator:
	  | ident IN range { RangeIterator($1, $3) }
	  | ident IN expr { ArrayIterator($1, $3) }

	range:
	  | expr COLON expr COLON expr { Range($1, $3, $5) }
	  | expr COLON expr { Range($1, $3, IntLit(1l)) }
	  | COLON expr COLON expr { Range(IntLit(0l), $2, $4) }
	  | COLON expr { Range(IntLit(0l), $2, IntLit(1l)) }

	top_level_statement:
	  | datatype ident LPAREN param_list RPAREN LCURLY statement_seq RCURLY
	      { FunctionDecl(false, $1, $2, $4, $7) }
	  | DEVICE datatype ident LPAREN param_list RPAREN LCURLY statement_seq RCURLY
	      { FunctionDecl(true, $2, $3, $5, $8) }
	  | datatype ident LPAREN param_list RPAREN SC
	      { ForwardDecl(false, $1, $2, $4) }
	  | DEVICE datatype ident LPAREN param_list RPAREN SC
	      { ForwardDecl(true, $2, $3, $5) }
	  | decl { Declaration($1) }

	param:
	  | datatype ident { PrimitiveDecl($1, $2) }
	  | datatype ident LSQUARE RSQUARE
	      { ArrayDecl($1, $2, []) }

	non_empty_param_list:
	  | param COMMA non_empty_param_list { $1 :: $3 }
	  | param { [$1] }

	param_list:
	  | non_empty_param_list { $1 }
	  | { [] }

	top_level:
	  | top_level_statement top_level {$1 :: $2}
	  | top_level_statement { [$1] }

	statement_seq:
	  | statement statement_seq {$1 :: $2 }
	  | { [] }

	%%

###A.4 environment.ml

	open Ast
	open Symgen
	(*
	 * When a new scope is entered, we push to the VariableMap stack; when a scope
	 * is left, we pop.
	 * *)

	(* Maps variable idents to type *)
	module VariableMap = Map.Make(struct type t = ident let compare = compare end);;

	(* Maps function idents to return type *)
	module FunctionMap = Map.Make(struct type t = ident let compare = compare end);;

	module FunctionDeclarationMap = Map.Make(struct type t = ident let compare = compare end);;

	exception Already_declared
	exception Invalid_environment
	exception Invalid_operation
	exception Not_implemented

	(* Kernel invocation functions are functions that invoke the kernel. The
	 * kernel_invoke_sym refers to the function name of this kernel invocation
	 * function, while the kernel_sym refers to the function name of the kernel
	 * function being invoked.
	 *)
	type kernel_invocation_function = {
	  kernel_invoke_sym: string;
	  kernel_sym: string;
	  func_type: string;
	  higher_order_func: ident;
	  };;

	(* Kernel functions are the functions that are executed on the GPU.
	 * The function_name is the name of the function in the vector code
	 * that is being called, and kernel_symbol is the function name of the
	 * generated global function that the kernel is compiled to (it needs to
	 * have an additional __global__ directive and operate on arrays, which is
	 * why it is a seperate function.  The hof is the type of higher-order-function
	 * being invoked.
	 *)
	type kernel_function = {
	  function_name: string;
	  hof: ident;
	  kernel_symbol: string;
	  function_type: string;
	};;

	type pfor_kernel = {
	    pfor_kernel_name: string;
	    pfor_iterators: iterator list;
	    pfor_arguments: decl list;
	    pfor_statement: statement;
	};;

	type 'a env = {
	  kernel_invocation_functions: kernel_invocation_function list;
	  kernel_functions: kernel_function list;
	  func_type_map: (bool * datatype * datatype list) FunctionMap.t;
	  scope_stack: datatype VariableMap.t list;
	  pfor_kernels: pfor_kernel list;
	  on_gpu: bool;
	}

	type 'a sourcecomponent =
	  | Verbatim of string
	  | Generator of ('a -> (string * 'a))
	  | NewScopeGenerator of ('a -> (string * 'a))

	let create =
	  {
	    kernel_invocation_functions = [];
	    kernel_functions = [];
	    func_type_map = FunctionMap.empty;
	    scope_stack = VariableMap.empty ::[];
	    pfor_kernels = [];
	    on_gpu = false;
	  }

	let update_env kernel_invocation_functions kernel_functions
	  func_type_map var_map_list pfor_kernels on_gpu =
	    {
	      kernel_invocation_functions = kernel_invocation_functions;
	      kernel_functions = kernel_functions;
	      func_type_map = func_type_map;
	      scope_stack = var_map_list;
	      pfor_kernels = pfor_kernels;
	      on_gpu = on_gpu;
	    }

	let var_in_scope ident env =
	    let rec check_scope scopes =
	        match scopes with
	          | [] -> false
	          | scope :: tail ->
	                if VariableMap.mem ident scope then
	                    true
	                else check_scope tail
	    in check_scope env.scope_stack

	let get_var_type ident env =
	  let rec check_scope scopes =
	    match scopes with
	     | [] -> raise Not_found
	     | scope :: tail ->
	         if VariableMap.mem ident scope then
	           VariableMap.find ident scope
	         else
	           check_scope tail in
	  let scope_stack = env.scope_stack in
	  check_scope scope_stack

	let is_var_declared ident env =
	  match env.scope_stack with
	   | [] -> false
	   | scope :: tail -> VariableMap.mem ident scope

	let set_var_type ident datatype env =
	  let scope, tail = (match env.scope_stack with
	                | scope :: tail -> scope, tail
	                | [] -> raise Invalid_environment) in
	  let new_scope = VariableMap.add ident datatype scope in
	  update_env env.kernel_invocation_functions env.kernel_functions
	    env.func_type_map (new_scope :: tail) env.pfor_kernels env.on_gpu

	let update_scope ident datatype (str, env) =
	  if is_var_declared ident env then
	    raise Already_declared
	  else
	    (str, set_var_type ident datatype env)

	let push_scope env =
	  update_env env.kernel_invocation_functions env.kernel_functions
	     env.func_type_map
	    (VariableMap.empty :: env.scope_stack) env.pfor_kernels env.on_gpu

	let pop_scope env =
	  match env.scope_stack with
	   | local_scope :: tail ->
	      update_env env.kernel_invocation_functions env.kernel_functions
	         env.func_type_map tail env.pfor_kernels env.on_gpu
	   | [] -> raise Invalid_environment

	let get_func_info ident env =
	    FunctionMap.find ident env.func_type_map

	let is_func_declared ident env =
	  FunctionMap.mem ident env.func_type_map

	let update_global_funcs function_type kernel_invoke_sym function_name hof kernel_sym (str, env) =
	  let new_kernel_funcs = {
	    kernel_invoke_sym = kernel_invoke_sym;
	    higher_order_func = hof; 
	    kernel_sym = kernel_sym;
	    func_type = function_type;
	  } :: env.kernel_invocation_functions in

	  let new_global_funcs = {
	    function_name = function_name;
	    hof = hof;
	    kernel_symbol = kernel_sym;
	    function_type = function_type;
	  } :: env.kernel_functions in

	  (str, update_env new_kernel_funcs new_global_funcs env.func_type_map 
	      env.scope_stack env.pfor_kernels env.on_gpu)

	let update_pfor_kernels kernel_name iterators arguments statement (str, env) =
	    let new_pfor_kernels = {
	        pfor_kernel_name = kernel_name;
	        pfor_iterators = iterators;
	        pfor_arguments = arguments;
	        pfor_statement = statement;
	    } :: env.pfor_kernels in
	    (str, update_env env.kernel_invocation_functions env.kernel_functions
	            env.func_type_map env.scope_stack new_pfor_kernels env.on_gpu)

	let set_on_gpu env =
	    update_env env.kernel_invocation_functions env.kernel_functions
	        env.func_type_map env.scope_stack env.pfor_kernels true

	let set_func_type ident device returntype arg_list env =
	  let new_func_type_map = FunctionMap.add ident (device, returntype, arg_list) env.func_type_map in
	  update_env env.kernel_invocation_functions env.kernel_functions
	    new_func_type_map env.scope_stack env.pfor_kernels env.on_gpu

	let update_functions ident device returntype arg_list (str, env) =
	  if is_func_declared ident env then
	    raise Already_declared
	  else
	    (str, set_func_type ident device returntype arg_list env)

	let combine initial_env components =
	    let f (str, env) component =
	        match component with
	         | Verbatim(verbatim) -> str ^ verbatim, env
	         | Generator(gen) ->
	           let new_str, new_env = gen env in
	            str ^ new_str, new_env
	         | NewScopeGenerator(gen) ->
	           let new_str, new_env = gen (push_scope env) in
	            str ^ new_str, pop_scope new_env in
	    List.fold_left f ("", initial_env) components

###A.5 symgen.ml

	let sym_num = ref 0;;

	let gensym () =
	    let sym = "__sym" ^ string_of_int !sym_num ^ "_" in
	        sym_num := (!sym_num + 1); sym

###A.6 detect.ml

	open Environment
	open Ast

	exception Not_allowed_on_gpu

	module IdentSet = Set.Make(
	  struct
	    let compare = (
	      let ident_to_str = function Ident(s) -> s
	      in (fun i1 i2 -> Pervasives.compare (ident_to_str i1) (ident_to_str i2))
	    )
	    type t = ident
	  end
	);;

	let combine_detect_tuples tuplist =
	    let combine_pair tup1 tup2 =
	        let (in1, out1) = tup1 and
	            (in2, out2) = tup2 in
	        (IdentSet.union in1 in2, IdentSet.union out1 out2) in
	    List.fold_left combine_pair (IdentSet.empty, IdentSet.empty) tuplist;;

	let rec detect_statement stmt env =
	    match stmt with
	      | CompoundStatement(slst) -> detect_statement_list slst env
	      | Declaration(decl) -> detect_decl decl env
	      | Expression(expr) -> detect_expr expr env
	      | IfStatement(expr, stmt1, stmt2) -> combine_detect_tuples
	            [detect_expr expr env; detect_statement stmt1 env;
	             detect_statement stmt2 env]
	      | WhileStatement(expr, stmt) -> combine_detect_tuples
	            [detect_expr expr env; detect_statement stmt env]
	      | ForStatement(iter_list, stmt) -> combine_detect_tuples
	            [detect_iter_list iter_list env; detect_statement stmt env]
	      | _ -> raise Not_allowed_on_gpu
	and detect_decl decl env =
	    match decl with
	      | AssigningDecl(ident, expr) -> detect_expr expr env
	      | _ -> raise Not_allowed_on_gpu
	and detect_expr expr env =
	    match expr with
	      | Binop(expr1, _, expr2) -> combine_detect_tuples
	            [detect_expr expr1 env; detect_expr expr2 env]
	      | AssignOp(lvalue, _, expr) -> combine_detect_tuples
	            [detect_lvalue lvalue true true env ; detect_expr expr env]
	      | Unop(_, expr) -> detect_expr expr env
	      | PostOp(lvalue, _) -> detect_lvalue lvalue true true env
	      | Assign(lvalue, expr) -> combine_detect_tuples
	            [detect_lvalue lvalue false true env; detect_expr expr env]
	      | Cast(_, expr) -> detect_expr expr env
	      | FunctionCall(ident, elist) ->
	            detect_expr_list elist env
	      | Lval(lvalue) -> detect_lvalue lvalue true false env
	      | _ -> (IdentSet.empty, IdentSet.empty)
	and detect_expr_list elist env =
	    let tuplist = List.map (fun expr -> detect_expr expr env) elist in
	    combine_detect_tuples tuplist
	and detect_statement_list slst env =
	    let tuplist = List.map (fun stmt -> detect_statement stmt env) slst in
	    combine_detect_tuples tuplist
	and detect_lvalue lvalue ins outs env =
	    match lvalue with
	      | Variable(ident) ->
	            if var_in_scope ident env then
	                if outs then
	                    (IdentSet.empty, IdentSet.empty)
	                else if ins then
	                    (IdentSet.singleton ident, IdentSet.empty)
	                else (IdentSet.empty, IdentSet.empty)
	            else (IdentSet.empty, IdentSet.empty)
	      | ArrayElem(ident, indices) ->
	            let (indins, indouts) as indtups =
	                detect_expr_list indices env in
	            if var_in_scope ident env then
	                ((if ins then IdentSet.add ident indins else indins),
	                 (if outs then IdentSet.add ident indouts else indouts))
	            else indtups
	      | ComplexAccess(expr, _) ->
	            (match expr with
	              | Lval(Variable(ident)) ->
	                    if var_in_scope ident env then
	                        ((if ins then IdentSet.singleton ident else IdentSet.empty),
	                         (if outs then IdentSet.singleton ident else IdentSet.empty))
	                    else (IdentSet.empty, IdentSet.empty)
	              | _ -> detect_expr expr env)
	and detect_iter iter env =
	    match iter with
	      | RangeIterator(_, range) ->
	            detect_range range env
	      | ArrayIterator(_, expr) ->
	            detect_expr expr env
	and detect_iter_list iter_list env =
	    let tuplist = List.map (fun iter -> detect_iter iter env) iter_list in
	    combine_detect_tuples tuplist
	and detect_range range env =
	    let Range(expr1, expr2, expr3) = range in
	    combine_detect_tuples
	        [detect_expr expr1 env; detect_expr expr2 env; detect_expr expr3 env]

	let detect stmt env =
	    let gpu_inputs, gpu_outputs = detect_statement stmt env in
	    (IdentSet.elements gpu_inputs, IdentSet.elements gpu_outputs)

	let dedup lst =
	    IdentSet.elements (List.fold_left 
	        (fun set itm -> IdentSet.add itm set) IdentSet.empty lst)

###A.7 generator.ml

	open Ast
	open Complex
	open Environment
	open Symgen
	open Detect

	module StringMap = Map.Make(String);;

	exception Unknown_type
	exception Empty_list
	exception Type_mismatch of string
	exception Not_implemented (* this should go away *)
	exception Invalid_operation

	let rec infer_type expr env =
	    let f type1 type2 =
	        match type1 with
	         | Some(t) -> (if t = type2 then Some(t)
	                        else raise (Type_mismatch "wrong type in list"))
	         | None -> Some(type2) in
	    let match_type expr_list =
	      let a = List.fold_left f None expr_list in
	        match a with
	         | Some(t) -> t
	         | None -> raise Empty_list in
	    match expr with
	      | Binop(expr1, op, expr2) -> (match op with
	         | LogAnd | LogOr | Eq | NotEq | Less | LessEq | Greater | GreaterEq ->
	            Bool
	         | _ -> match_type [infer_type expr1 env; infer_type expr2 env])
	      | CharLit(_) -> Char
	      | ComplexLit(re, im) ->
	          (match (infer_type re env), (infer_type im env) with
	            | (Float32, Float32) -> Complex64
	            | (Float64, Float64) -> Complex128
	            | (t1, t2) -> raise (Type_mismatch "expected complex"))
	      | FloatLit(_) -> Float64
	      | Int64Lit(_) -> Int64
	      | IntLit(_) -> Int32
	      | StringLit(_) -> String
	      | ArrayLit(exprs) ->
	          let f expr = infer_type expr env in
	          ArrayType(match_type (List.map f exprs))
	      | Cast(datatype, expr) -> datatype
	      | Lval(lval) -> (match lval with
	          | ArrayElem(ident, _) ->
	                (match infer_type (Lval(Variable(ident))) env with
	                  | ArrayType(t) -> t
	                  | _ -> raise (Type_mismatch "Cannot access element of non-array"))
	          | Variable(i) -> Environment.get_var_type i env
	          | ComplexAccess(expr1,ident) ->
	              (match (infer_type expr1 env) with
	                | Complex64 -> Float32
	                | Complex128 -> Float64
	                | _ -> raise Invalid_operation))
	      | AssignOp(lval, _, expr) ->
	            let l = Lval(lval) in
	            match_type [infer_type l env; infer_type expr env]
	      | Unop(op, expr) -> (match op with
	            LogNot -> Bool
	          | _ -> infer_type expr env
	        )
	      | PostOp(lval, _) -> let l = Lval(lval) in infer_type l env
	      | Assign(lval, expr) ->
	            let l = Lval(lval) in
	            match_type [infer_type l env; infer_type expr env]
	      | FunctionCall(i, es) -> (match i with
	          | Ident("len") | Ident("random") -> Int32
	          | Ident("printf") | Ident("inline") | Ident("assert") -> Void
	          | Ident("time") -> Float64
	          | Ident("abs") -> (match es with
	              | [expr] -> (match infer_type expr env with
	                  | Complex64 -> Float32
	                  | Complex128 -> Float64
	                  | t -> t)
	              | _ -> raise (Type_mismatch "Wrong number of arguments to abs()"))
	          | _ -> let (_,dtype,_) = Environment.get_func_info i env in dtype)

	      | HigherOrderFunctionCall(hof, f, expr) ->
	          (match(hof) with
	            | Ident("map") -> ArrayType(let (_,dtype,_) = Environment.get_func_info f env
	                  in dtype)
	            | Ident("reduce") -> let (_,dtype,_) = Environment.get_func_info f env in dtype
	            | _ -> raise Invalid_operation)

	let generate_ident ident env =
	  match ident with
	    Ident(s) -> Environment.combine env [Verbatim(s)]

	let rec generate_datatype datatype env =
	    match datatype with
	      | Bool -> Environment.combine env [Verbatim("bool")]
	      | Char -> Environment.combine env [Verbatim("char")]
	      | Int8 -> Environment.combine env [Verbatim("int8_t")]
	      | UInt8 -> Environment.combine env [Verbatim("uint8_t")]
	      | Int16 -> Environment.combine env [Verbatim("int16_t")]
	      | UInt16 -> Environment.combine env [Verbatim("uint16_t")]
	      | Int32 -> Environment.combine env [Verbatim("int32_t")]
	      | UInt32 -> Environment.combine env [Verbatim("uint32_t")]
	      | Int64 -> Environment.combine env [Verbatim("int64_t")]
	      | UInt64 -> Environment.combine env [Verbatim("uint64_t")]
	      | Float32 -> Environment.combine env [Verbatim("float")]
	      | Float64 -> Environment.combine env [Verbatim("double")]

	      | Complex64 -> Environment.combine env [Verbatim("cuFloatComplex")]
	      | Complex128 -> Environment.combine env [Verbatim("cuDoubleComplex")]

	      | String -> Environment.combine env [Verbatim("char *")]

	      | ArrayType(t) -> Environment.combine env [
	            Verbatim("VectorArray<");
	            Generator(generate_datatype t);
	            Verbatim(">")
	        ]

	      | _ -> raise Unknown_type

	let generate_rettype dtype env =
	    match dtype with
	      | Void -> "void", env
	      | _ -> generate_datatype dtype env

	let rec generate_lvalue modify lval env =
	    let rec generate_array_index array_id dim exprs env =
	        let generate_mid_index array_id dim expr env =
	            let typ = match (Environment.get_var_type array_id env) with
	              | ArrayType(typ) -> typ
	              | _ -> raise (Type_mismatch "Cannot index into non-array") in
	            Environment.combine env [
	                Verbatim("get_mid_index<");
	                Generator(generate_datatype typ);
	                Verbatim(">(");
	                Generator(generate_ident array_id);
	                Verbatim(", ");
	                Generator(generate_expr expr);
	                Verbatim(", " ^ string_of_int dim ^ ")");
	            ] in
	    match exprs with
	      | [] -> raise Invalid_operation
	      | [expr] ->
	            generate_mid_index array_id dim expr env
	      | expr :: tail ->
	            Environment.combine env [
	                Generator(generate_mid_index array_id dim expr);
	                Verbatim("+");
	                Generator(generate_array_index array_id (dim + 1) tail)
	            ] in
	  match lval with
	    | Variable(i) ->
	        Environment.combine env [Generator(generate_ident i)]
	    | ArrayElem(ident, es) ->
	        if env.on_gpu then
	            Environment.combine env [
	                Generator(generate_ident ident);
	                Verbatim("->values[");
	                Generator(generate_array_index ident 0 es);
	                Verbatim("]")
	            ]
	        else
	            Environment.combine env [
	              Generator(generate_ident ident);
	              Verbatim(".elem(" ^ (if modify then "true" else "false") ^ ", ");
	              Generator(generate_expr_list es);
	              Verbatim(")")
	            ]
	     | ComplexAccess(expr, ident) -> (
	         let _op = match ident with
	           Ident("re") -> ".x"
	         | Ident("im") -> ".y"
	         | _ -> raise Not_found in
	           Environment.combine env [
	             Generator(generate_expr expr);
	             Verbatim(_op)
	           ]
	       )
	and generate_expr expr env =
	  match expr with
	    Binop(e1,op,e2) ->
	      let datatype = (infer_type e1 env) in
	      (match datatype with
	        | Complex64 | Complex128 ->
	            let func = match op with
	              | Add -> "cuCadd"
	              | Sub -> "cuCsub"
	              | Mul -> "cuCmul"
	              | Div -> "cuCdiv"
	              | _ -> raise Invalid_operation in
	            let func = if datatype == Complex128 then
	                func else func ^ "f" in
	            Environment.combine env [
	                Verbatim(func ^ "(");
	                Generator(generate_expr e1);
	                Verbatim(",");
	                Generator(generate_expr e2);
	                Verbatim(")")
	            ]
	       | _ ->
	        let _op = match op with
	          | Add -> "+"
	          | Sub -> "-"
	          | Mul -> "*"
	          | Div -> "/"
	          | Mod -> "%"
	          | Lshift -> "<<"
	          | Rshift -> ">>"
	          | Less -> "<"
	          | LessEq -> "<="
	          | Greater -> ">"
	          | GreaterEq -> ">="
	          | Eq -> "=="
	          | NotEq -> "!="
	          | BitAnd -> "&"
	          | BitXor -> "^"
	          | BitOr -> "|"
	          | LogAnd -> "&&"
	          | LogOr -> "||"
	        in
	        Environment.combine env [
	          Verbatim("(");
	          Generator(generate_expr e1);
	          Verbatim(") " ^ _op ^ " (");
	          Generator(generate_expr e2);
	          Verbatim(")");
	        ])

	  | AssignOp(lvalue, op, e) ->
	      (* change lval op= expr to lval = lval op expr *)
	      generate_expr (Assign(lvalue, Binop(Lval(lvalue), op, e))) env
	  | Unop(op,e) -> (
	      let simple_unop _op e = Environment.combine env [
	          Verbatim(_op);
	          Generator(generate_expr e)
	      ] in
	      match op with
	          Neg -> simple_unop "-" e
	        | LogNot -> simple_unop "!" e
	        | BitNot -> simple_unop "~" e
	    )
	  | PostOp(lvalue, op) -> (
	      let _op = match op with
	          Dec -> "--"
	        | Inc -> "++"
	      in
	      Environment.combine env [
	        Generator(generate_lvalue true lvalue);
	        Verbatim(_op)
	      ]
	    )
	  | Assign(lvalue, e) ->
	      Environment.combine env [
	        Generator(generate_lvalue true lvalue);
	        Verbatim(" = ");
	        Generator(generate_expr e)
	      ]
	  | IntLit(i) ->
	      Environment.combine env [Verbatim(Int32.to_string i)]
	  | Int64Lit(i) ->
	      Environment.combine env [Verbatim(Int64.to_string i)]
	  | FloatLit(f) ->
	      Environment.combine env [Verbatim(string_of_float f)]
	  | ComplexLit(re, im) as lit ->
	      let make_func = (match (infer_type lit env) with
	        | Complex64 -> "make_cuFloatComplex"
	        | Complex128 -> "make_cuDoubleComplex"
	        | t -> raise (Type_mismatch "Mismatch in ComplexLit")) in
	      Environment.combine env [
	        Verbatim(make_func ^ "(");
	        Generator(generate_expr re);
	        Verbatim(", ");
	        Generator(generate_expr im);
	        Verbatim(")")
	      ]
	  | StringLit(s) ->
	      Environment.combine env [Verbatim("\"" ^ s ^ "\"")]
	  | CharLit(c) ->
	      Environment.combine env [
	        Verbatim("'" ^ Char.escaped c ^ "'")
	      ]
	  | ArrayLit(es) as lit ->
	      let typ = (match (infer_type lit env) with
	       | ArrayType(t) -> t
	       | t -> raise (Type_mismatch "ArrayLit")) in
	      let len = List.length es in
	      Environment.combine env [
	          Verbatim("VectorArray<");
	          Generator(generate_datatype typ);
	          Verbatim(">(1, (size_t) " ^ string_of_int len ^ ")");
	          Generator(generate_array_init_chain 0 es)
	      ]
	  | Cast(d,e) ->
	      Environment.combine env [
	        Verbatim("(");
	        Generator(generate_datatype d);
	        Verbatim(") (");
	        Generator(generate_expr e);
	        Verbatim(")")
	      ]
	  | FunctionCall(i,es) ->
	      Environment.combine env (match i with
	        | Ident("inline") -> (match es with
	            | StringLit(str) :: [] -> [Verbatim(str)]
	            | _ -> raise (Type_mismatch "expected string"))
	        | Ident("printf") -> [
	            Verbatim("printf(");
	            Generator(generate_expr_list es);
	            Verbatim(")");
	          ]
	        | Ident("len") -> (match es with
	            | expr :: [] -> (match (infer_type expr env) with
	                | ArrayType(_) -> [
	                    Verbatim("(");
	                    Generator(generate_expr expr);
	                    Verbatim(").size()")
	                ]
	                | String -> [
	                    Verbatim("strlen(");
	                    Generator(generate_expr expr);
	                    Verbatim(")")
	                ]
	                | _ -> raise (Type_mismatch "cannot compute length"))
	            | expr1 :: expr2 :: [] ->
	                (match (infer_type expr1 env), (infer_type expr2 env) with
	                  | (ArrayType(_), Int32) -> [
	                      Verbatim("(");
	                      Generator(generate_expr expr1);
	                      Verbatim(").length(");
	                      Generator(generate_expr expr2);
	                      Verbatim(")");
	                  ]
	                  | _ -> raise (Type_mismatch
	                      "len() with two arguments must have types array and int"))
	            | _ -> raise (Type_mismatch "incorrect number of parameters"))
	        | Ident("assert") -> (match es with
	            | expr :: [] -> (match infer_type expr env with
	                | Bool -> [
	                    Verbatim("if (!(");
	                    Generator(generate_expr expr);
	                    Verbatim(")) {printf(\"Assertion failed: ");
	                    Generator(generate_expr expr);
	                    Verbatim("\"); exit(EXIT_FAILURE); }");
	                ]
	                | _ -> raise (Type_mismatch "Cannot assert on non-boolean expression"))
	            | _ -> raise (Type_mismatch "too many parameters"))
	        | Ident("random") -> (match es with
	            | [] -> [ Verbatim("random()") ]
	            | _ -> raise (Type_mismatch "Too many argument to random"))
	        | Ident("abs") -> (match es with
	            | [expr] ->
	                let absfunc = (match infer_type expr env with
	                  | Int8 | Int16 | Int32 | Int64 | Float32 | Float64 -> "abs"
	                  | Complex64 -> "cuCabsf"
	                  | Complex128 -> "cuCabs"
	                  | _ -> raise (Type_mismatch "abs() takes only numbers")) in
	                [
	                    Verbatim(absfunc ^ "(");
	                    Generator(generate_expr expr);
	                    Verbatim(")");
	                ]
	            | _ -> raise (Type_mismatch "Wrong number of arguments to abs()"))
	        | Ident("time") -> (match es with
	            | [] -> [ Verbatim("get_time()") ]
	            | _ -> raise (Type_mismatch "time() takes no arguments"))
	        | _ -> let _, _, arg_list = Environment.get_func_info i env in
	            let rec validate_type_list types expressions env =
	                match (types, expressions) with
	                  | typ :: ttail, expr :: etail ->
	                        if (typ = infer_type expr env) then
	                            validate_type_list ttail etail env
	                        else false
	                  | [], [] -> true
	                  | [], _ -> false
	                  | _, [] -> false in
	            if (validate_type_list arg_list es env) then [
	                Generator(generate_ident i);
	                Verbatim("(");
	                Generator(generate_expr_list es);
	                Verbatim(")")
	            ] else raise (Type_mismatch "Arguments of wrong type"))

	  | HigherOrderFunctionCall(i1,i2,es) ->
	      (match infer_type es env with
	        | ArrayType(function_type) ->
	            let function_type_str, _ = generate_datatype function_type env in
	            let kernel_invoke_sym = Symgen.gensym () in
	            let kernel_sym = Symgen.gensym () in
	            let function_name, _ = generate_ident i2 env in
	            Environment.update_global_funcs function_type_str kernel_invoke_sym 
	                function_name i1 kernel_sym (Environment.combine env [
	                    Verbatim(kernel_invoke_sym ^ "(");
	                    Generator(generate_expr es);
	                    Verbatim(")")])
	        | t -> let dtype, _ = generate_datatype t env in
	            raise (Type_mismatch
	                    ("Expected array as argument to HOF, got " ^ dtype)))
	  | Lval(lvalue) ->
	      Environment.combine env [Generator(generate_lvalue false lvalue)]
	and generate_array_init_chain ind expr_list env =
	    match expr_list with
	      | [] -> "", env
	      | expr :: tail ->
	            Environment.combine env [
	                Verbatim(".chain_set(" ^ string_of_int ind ^ ", ");
	                Generator(generate_expr expr);
	                Verbatim(")");
	                Generator(generate_array_init_chain (ind + 1) tail)
	            ]
	and generate_expr_list expr_list env =
	  match expr_list with
	   | [] -> Environment.combine env []
	   | lst ->
	       Environment.combine env [Generator(generate_nonempty_expr_list lst)]
	and generate_nonempty_expr_list expr_list env =
	  match expr_list with
	   | expr :: [] ->
	       Environment.combine env [Generator(generate_expr expr)]
	   | expr :: tail ->
	       Environment.combine env [
	         Generator(generate_expr expr);
	         Verbatim(", ");
	         Generator(generate_nonempty_expr_list tail)
	       ]
	   | [] -> raise Empty_list
	and generate_decl decl env =
	  match decl with
	    | AssigningDecl(ident,e) ->
	        let datatype = (infer_type e env) in
	        Environment.update_scope ident datatype
	            (Environment.combine env [
	              Generator(generate_datatype datatype);
	              Verbatim(" ");
	              Generator(generate_ident ident);
	              Verbatim(" = ");
	              Generator(generate_expr e)
	            ])
	    | PrimitiveDecl(d,i) ->
	        Environment.update_scope i d (Environment.combine env [
	          Generator(generate_datatype d);
	          Verbatim(" ");
	          Generator(generate_ident i)
	        ])
	    | ArrayDecl(d,i,es) ->
	        Environment.update_scope i (ArrayType(d))
	            (if env.on_gpu then (match es with
	              | [] -> Environment.combine env [
	                  Verbatim("device_info<");
	                  Generator(generate_datatype d);
	                  Verbatim("> *");
	                  Generator(generate_ident i);
	              ]
	              | _ -> raise Not_allowed_on_gpu)
	            else (match es with
	              | [] -> Environment.combine env [
	                  Verbatim("VectorArray<");
	                  Generator(generate_datatype d);
	                  Verbatim("> ");
	                  Generator(generate_ident i);
	              ]
	              | _ ->
	                  Environment.combine env [
	                    Verbatim("VectorArray<");
	                    Generator(generate_datatype d);
	                    Verbatim("> ");
	                    Generator(generate_ident i);
	                    Verbatim("(" ^ string_of_int (List.length es) ^ ", ");
	                    Generator(generate_expr_list es);
	                    Verbatim(")")
	                  ]))

	let generate_range range env =
	  match range with
	   | Range(e1,e2,e3) ->
	      Environment.combine env [
	        Verbatim("range(");
	        Generator(generate_expr e1);
	        Verbatim(", ");
	        Generator(generate_expr e2);
	        Verbatim(", ");
	        Generator(generate_expr e3);
	        Verbatim(")")
	      ]

	let generate_iterator iterator env =
	  match iterator with
	   | RangeIterator(i, r) ->
	       Environment.combine env [
	         Generator(generate_ident i);
	         Verbatim(" in ");
	         Generator(generate_range r)
	       ]
	   | ArrayIterator(i, e) ->
	       Environment.combine env [
	         Generator(generate_ident i);
	         Verbatim(" in ");
	         Generator(generate_expr e)
	       ]

	let rec generate_iterator_list iterator_list env =
	  match iterator_list with
	   | [] -> Environment.combine env [Verbatim("_")]
	   | iterator :: tail ->
	       Environment.combine env [
	         Generator(generate_iterator iterator);
	         Verbatim(", ");
	         Generator(generate_iterator_list tail)
	       ]

	let rec generate_nonempty_decl_list decl_list env =
	  match decl_list with
	   | decl :: [] ->
	       Environment.combine env [Generator(generate_decl decl)]
	   | decl :: tail ->
	       Environment.combine env [
	         Generator(generate_decl decl);
	         Verbatim(", ");
	         Generator(generate_nonempty_decl_list tail)
	       ]
	   | [] -> raise Empty_list

	let generate_decl_list decl_list env =
	  match decl_list with
	   | [] -> Environment.combine env [Verbatim("void")]
	   | (decl :: tail) as lst ->
	       Environment.combine env [
	         Generator(generate_nonempty_decl_list lst)
	       ]

	let rec generate_statement statement env =
	    match statement with
	      | CompoundStatement(ss) ->
	          Environment.combine env [
	            Verbatim("{\n");
	            NewScopeGenerator(generate_statement_list ss);
	            Verbatim("}\n")
	          ]
	      | Declaration(d) ->
	          Environment.combine env [
	            Generator(generate_decl d);
	            Verbatim(";")
	          ]
	      | Expression(e) ->
	          Environment.combine env [
	            Generator(generate_expr e);
	            Verbatim(";")
	          ]
	      | EmptyStatement ->
	          Environment.combine env [Verbatim(";")]
	      | IfStatement(e, s1, s2) ->
	          Environment.combine env [
	            Verbatim("if (");
	            Generator(generate_expr e);
	            Verbatim(")\n");
	            NewScopeGenerator(generate_statement s1);
	            Verbatim("\nelse\n");
	            NewScopeGenerator(generate_statement s2)
	          ]
	      | WhileStatement(e, s) ->
	          Environment.combine env [
	            Verbatim("while (");
	            Generator(generate_expr e);
	            Verbatim(")\n");
	            NewScopeGenerator(generate_statement s)
	          ]
	      | ForStatement(is, s) ->
	          Environment.combine env [
	            NewScopeGenerator(generate_for_statement (is, s));
	          ]
	      | PforStatement(is, s) ->
	          Environment.combine env [
	            NewScopeGenerator(generate_pfor_statement is s)
	          ]
	      | FunctionDecl(device, return_type, identifier, arg_list, body_sequence) ->
	          let str, env = Environment.combine env [
	              NewScopeGenerator(generate_function device return_type
	                                identifier arg_list body_sequence)
	            ] in
	          let types = List.map (function x ->
	            match x with
	            |PrimitiveDecl(t, id) -> t
	            |ArrayDecl(t, id, expr_list) -> ArrayType(t)
	            | _ -> raise Invalid_operation
	          ) arg_list in
	          let new_str, new_env = Environment.update_functions identifier
	            device return_type types (str, env) in
	          new_str, new_env

	      | ForwardDecl(device, return_type, ident, decl_list) ->
	            let types = List.map (function x ->
	              match x with
	              |PrimitiveDecl(t, id) -> t
	              |ArrayDecl(t, id, expr_list) -> ArrayType(t)
	              | _ -> raise Invalid_operation
	            ) decl_list in
	            Environment.update_functions ident device return_type types
	                (Environment.combine env [
	                    Verbatim(if device then "__device__ " else "");
	                    Generator(generate_datatype return_type);
	                    Verbatim(" ");
	                    Generator(generate_ident ident);
	                    Verbatim("(");
	                    Generator(generate_decl_list decl_list);
	                    Verbatim(");")
	                ])
	      | ReturnStatement(e) ->
	          Environment.combine env [
	            Verbatim("return ");
	            Generator(generate_expr e);
	            Verbatim(";")
	          ]
	      | VoidReturnStatement ->
	          Environment.combine env [Verbatim("return;")]

	and generate_statement_list statement_list env =
	    match statement_list with
	     | [] -> Environment.combine env []
	     | statement :: tail ->
	         Environment.combine env [
	           Generator(generate_statement statement);
	           Verbatim("\n");
	           Generator(generate_statement_list tail)
	         ]

	and generate_function device returntype ident params statements env =
	    Environment.combine env [
	        Verbatim(if device then "__device__ " else "");
	        Generator(generate_rettype returntype);
	        Verbatim(" ");
	        Generator(generate_ident ident);
	        Verbatim("(");
	        Generator(generate_decl_list params);
	        Verbatim(") {\n");
	        Generator(generate_statement_list statements);
	        Verbatim("}");
	    ]

	and generate_for_statement (iterators, statements) env =

	  let iter_name iterator = match iterator with
	    ArrayIterator(Ident(s),_) -> s
	  | RangeIterator(Ident(s),_) -> s
	  in

	  (* map iterators to their properties
	   * key on s rather than Ident(s) to save some effort... *)
	  let iter_map =

	    (* create symbols for an iterator's length and index.
	     *
	     * there is also a mod symbol - since we're flattening multiple
	     * iterators into a single loop, we need to know how often each one
	     * wraps around
	     *
	     * the output symbol is simply the symbol requested in the original
	     * vector code
	     *
	     * for ranges, we have start:_:inc
	     * for arrays, start = 0, inc = 1
	     *)
	    let get_iter_properties iterator =
	      let len_sym = Ident(Symgen.gensym () ^ iter_name iterator ^ "_len") in
	      let mod_sym = Ident(Symgen.gensym () ^ iter_name iterator ^ "_mod") in
	      let div_sym = Ident(Symgen.gensym () ^ iter_name iterator ^ "_div") in
	      let output_sym = match iterator with
	        ArrayIterator(i,_) -> i
	      | RangeIterator(i,_) -> i
	      in
	      (* start_sym and inc_sym are never actually used for array iterators...
	       * the only consequence is that the generated symbols in our output code
	       * will be non-consecutive because of these "wasted" identifiers *)
	      let start_sym = Ident(Symgen.gensym () ^ iter_name iterator ^ "_start") in
	      let inc_sym = Ident(Symgen.gensym () ^ iter_name iterator ^ "_inc") in
	      (iterator, len_sym, mod_sym, div_sym, output_sym, start_sym, inc_sym)
	    in

	    List.fold_left (fun m i -> StringMap.add (iter_name i) (get_iter_properties i) (m)) (StringMap.empty) (iterators)
	  in

	  (* generate code to calculate the length of each iterator
	   * and initialize the corresponding variables
	   *
	   * also calculate start and inc *)
	  let iter_length_initializers =

	    let iter_length_inits _ (iter, len_sym, _, _, _, start_sym, inc_sym) acc =
	      match iter with
	        ArrayIterator(_,e) -> [
	          Declaration(
	            AssigningDecl(len_sym, FunctionCall(Ident("len"), e :: [])))
	        ] :: acc
	      | RangeIterator(_,Range(start_expr,stop_expr,inc_expr)) -> (
	          (* the number of iterations in the iterator a:b:c is n, where
	           * n = (b-a-1) / c + 1 *)
	          let delta = Binop(stop_expr, Sub, Lval(Variable(start_sym))) in
	          let delta_fencepost = Binop(delta, Sub, IntLit(Int32.of_int 1)) in
	          let n = Binop(delta_fencepost, Div, Lval(Variable(inc_sym))) in
	          let len_expr = Binop(n, Add, IntLit(Int32.of_int 1)) in
	          [
	            Declaration(AssigningDecl(start_sym, start_expr));
	            Declaration(AssigningDecl(inc_sym, inc_expr));
	            Declaration(AssigningDecl(len_sym, len_expr));
	          ] :: acc
	        )
	    in

	    List.concat (StringMap.fold (iter_length_inits) (iter_map) ([]))
	  in

	  (* the total length of our for loop is the product
	   * of lengths of all iterators *)
	  let iter_max =
	    StringMap.fold (fun _ (_, len_sym, _, _, _, _, _) acc ->
	      Binop(acc, Mul, Lval(Variable(len_sym)))) (iter_map) (IntLit(Int32.of_int 1))
	  in

	  (* figure out how often each iterator wraps around
	   * the rightmost iterator wraps the fastest (i.e. mod its own length)
	   * all other iterators wrap modulo (their own length times the mod of
	   * the iterator directly to the right) *)
	  let iter_mod_initializers =
	    let iter_initializer iterator acc =
	      let name = iter_name iterator in
	      let mod_ident, div_ident, len_ident = match (StringMap.find name iter_map) with (_,l,m,d,_,_,_) -> m,d,l in
	      match acc with
	        [] -> [
	          Declaration(AssigningDecl(mod_ident, Lval(Variable(len_ident))));
	          Declaration(AssigningDecl(div_ident, IntLit(Int32.of_int 1)));
	        ]
	      | Declaration(AssigningDecl(prev_mod_ident, _)) :: _ -> (
	          List.append [
	            Declaration(AssigningDecl(mod_ident, Binop(Lval(Variable(len_ident)), Mul, Lval(Variable(prev_mod_ident)))));
	            Declaration(AssigningDecl(div_ident, Lval(Variable(prev_mod_ident))));
	          ] acc)
	      | _ -> [] 
	    in
	    (* we've built up the list with the leftmost iterator first,
	     * but we need the rightmost declared first due to dependencies.
	     * reverse it! *)
	    List.rev (List.fold_right (iter_initializer) (iterators) ([]))
	  in

	  (* initializers for the starting and ending value
	   * of the index we're generating *)
	  let iter_ptr_ident = Ident(Symgen.gensym () ^ "iter_ptr") in
	  let iter_max_ident = Ident(Symgen.gensym () ^ "iter_max") in
	  let bounds_initializers = [
	    Declaration(AssigningDecl(iter_ptr_ident, IntLit(Int32.of_int 0)));
	    Declaration(AssigningDecl(iter_max_ident, iter_max));
	  ] in

	  (* these assignments will occur at the beginning of each iteration *)
	  let output_assignments =
	    let iter_properties s = match (StringMap.find s iter_map) with
	      (_, _, mod_sym, div_sym, output_sym, start_sym, inc_sym) ->
	        (mod_sym, div_sym, output_sym, start_sym, inc_sym)
	    in
	    let idx mod_sym div_sym =
	      Binop(
	        Binop(Lval(Variable(iter_ptr_ident)), Mod, Lval(Variable(mod_sym))),
	        Div,
	        Lval(Variable(div_sym)))
	    in
	    let iter_assignment iterator = match iterator with
	      ArrayIterator(Ident(s), Lval(Variable(ident))) -> (
	        let mod_sym, div_sym, output_sym, _, _ = iter_properties s in
	        Declaration(AssigningDecl(output_sym, Lval(ArrayElem(ident,[idx mod_sym div_sym])))))
	    | RangeIterator(Ident(s),_) -> (
	        let mod_sym, div_sym, output_sym, start_sym, inc_sym = iter_properties s in
	        let offset = Binop(idx mod_sym div_sym, Mul, Lval(Variable(inc_sym))) in
	        let origin = Lval(Variable(start_sym)) in
	        let rhs = Binop(origin, Add, offset) in
	        Declaration(AssigningDecl(output_sym, rhs))
	      )
	    | _ -> raise Not_implemented
	    in
	    List.map (iter_assignment) (iterators)
	  in

	  Environment.combine env [
	    Verbatim("{\n");
	    Generator(generate_statement_list iter_length_initializers);
	    Generator(generate_statement_list iter_mod_initializers);
	    Generator(generate_statement_list bounds_initializers);
	    Verbatim("for (; ");
	    Generator(generate_expr (Binop(Lval(Variable(iter_ptr_ident)), Less, Lval(Variable(iter_max_ident)))));
	    Verbatim("; ");
	    Generator(generate_expr (PostOp(Variable(iter_ptr_ident), Inc)));
	    Verbatim(") {\n");
	    Generator(generate_statement_list output_assignments);
	    Generator(generate_statement statements);
	    Verbatim("}\n");
	    Verbatim("}\n");
	  ]
	and generate_pfor_statement iters stmt env =
	    (* generate intermediate symbols for array iterators *)
	    let rec get_array_ident_array ident_array index = function
	      | ArrayIterator(_, expr) :: tl ->
	            ident_array.(index) <- Symgen.gensym ();
	            get_array_ident_array ident_array (index + 1) tl
	      | _ :: tl -> get_array_ident_array ident_array (index + 1) tl
	      | [] -> ident_array in
	    let niters = List.length iters in
	    let array_ident_array =
	        get_array_ident_array (Array.make niters "") 0 iters in
	    (* setup an array of iterator structs *)
	    let gen_struct_mem iter_arr index mem_name expr env =
	        if (infer_type expr env) == Int32 then
	            Environment.combine env [
	                Verbatim(iter_arr ^ "[" ^ string_of_int index ^ "]."
	                            ^ mem_name ^ " = ");
	                Generator(generate_expr expr);
	                Verbatim(";\n")
	            ]
	        else raise (Type_mismatch "Iterator control must have type int32") in
	    (* assign start, stop, and inc for each iterator *)
	    let rec gen_iter_struct iter_arr index iters env =
	        match iters with
	          | RangeIterator(_, Range(start_expr, stop_expr, inc_expr)) :: tl ->
	                Environment.combine env [
	                    Generator(gen_struct_mem iter_arr index "start" start_expr);
	                    Generator(gen_struct_mem iter_arr index "stop" stop_expr);
	                    Generator(gen_struct_mem iter_arr index "inc" inc_expr);
	                    Generator(gen_iter_struct iter_arr (index + 1) tl)
	                ]
	          | ArrayIterator(_, array_expr) :: tl ->
	                let array_sym = Ident(array_ident_array.(index)) in
	                Environment.combine env [
	                    Generator(generate_decl (AssigningDecl(
	                        array_sym, array_expr)));
	                    Verbatim(";\n");
	                    Generator(gen_struct_mem iter_arr index "start" (IntLit(0l)));
	                    Generator(gen_struct_mem iter_arr index "stop"
	                        (FunctionCall(Ident("len"),
	                            [Lval(Variable(array_sym))])));
	                    Generator(gen_struct_mem iter_arr index "inc" (IntLit(1l)));
	                    Generator(gen_iter_struct iter_arr (index + 1) tl)
	                ]
	          | [] -> "", env in
	    (* turn the array into a list *)
	    let rec get_array_ident_list cur_list ident_array index n =
	        if index < n then
	            match ident_array.(index) with
	                (* empty strings are from range iters, so ignore them *)
	              | "" -> get_array_ident_list cur_list ident_array (index + 1) n
	              | id -> get_array_ident_list (Ident(id) :: cur_list) ident_array
	                            (index + 1) n
	        else cur_list in
	    (* array arguments must have their device info pointer passed in
	     * everything else can be passed in as-is *)
	    let generate_kernel_arg env id =
	        let Ident(s) = id in
	        match get_var_type id env with
	          | ArrayType(_) -> s ^ ".devInfo()"
	          | _ -> s in
	    let generate_ident_list ident_list env =
	        let ident_str = String.concat ", " (List.map
	                            (generate_kernel_arg env) ident_list) in
	        (* put a leading comma if non-empty, otherwise just return "" *)
	        if ident_str = "" then "", env else (", " ^ ident_str), env in
	    let generate_output_markings output_list =
	        String.concat "" (List.map (function Ident(s) ->
	                                    s ^ ".markDeviceDirty();\n") output_list) in
	    let iter_arr = Symgen.gensym () and
	        iter_devptr = Symgen.gensym () and
	        total_iters = Symgen.gensym () and
	        kernel_name = Symgen.gensym () and
	        gpu_inputs, gpu_outputs = detect stmt env and
	        array_ident_list = get_array_ident_list [] array_ident_array 0 niters in
	    let full_ident_list =
	        Detect.dedup (gpu_outputs @ gpu_inputs @ array_ident_list) in
	    let gen_kernel_decl id =
	        match Environment.get_var_type id env with
	          | ArrayType(typ) -> ArrayDecl(typ, id, [])
	          | typ -> PrimitiveDecl(typ, id) in
	    let kernel_args = List.map gen_kernel_decl full_ident_list in
	    Environment.update_pfor_kernels kernel_name iters kernel_args stmt
	        (Environment.combine env [
	            Verbatim("{\nstruct range_iter " ^ iter_arr ^
	                "[" ^ string_of_int niters ^ "];\n");
	            Generator(gen_iter_struct iter_arr 0 iters);
	            Verbatim("fillin_iters(" ^ iter_arr ^ ", " ^
	                        string_of_int niters ^ ");\n");
	            Verbatim("struct range_iter *" ^ iter_devptr ^
	                        " = device_iter(" ^ iter_arr ^ ", " ^
	                        string_of_int niters ^ ");\n");
	            Verbatim("size_t " ^ total_iters ^ " = total_iterations(" ^
	                        iter_arr ^ ", " ^ string_of_int niters ^ ");\n");
	            Verbatim(kernel_name ^ "<<<ceil_div(" ^ total_iters ^
	                        ", BLOCK_SIZE), BLOCK_SIZE>>>(" ^ iter_devptr ^ ", " ^
	                        string_of_int niters ^ ", " ^ total_iters);
	            Generator(generate_ident_list full_ident_list);
	            Verbatim(");\ncudaDeviceSynchronize();
	                        checkError(cudaGetLastError());\n");
	            Verbatim(generate_output_markings gpu_outputs);
	            Verbatim("cudaFree(" ^ iter_devptr ^ ");\n}\n")
	        ])

	let generate_toplevel tree =
	    let env = Environment.create in
	    Environment.combine env [
	        Generator(generate_statement_list tree);
	        Verbatim("\nint main(void) { return vec_main(); }\n")
	    ]

	let generate_kernel_invocation_functions env =
	  let rec generate_functions funcs str =
	    match funcs with
	    | [] -> str
	    | head :: tail ->
	        (match head.higher_order_func with
	          | Ident("map") ->
	            let new_str = str ^ "\nVectorArray<" ^ head.func_type ^ "> " ^ head.kernel_invoke_sym ^ "(" ^
	            "VectorArray<" ^ head.func_type ^ "> input){
	              int inputSize = input.size();
	              VectorArray<" ^ head.func_type ^ " > output = input.dim_copy();
	              " ^ head.kernel_sym ^
	              "<<<ceil_div(inputSize,BLOCK_SIZE),BLOCK_SIZE>>>(output.devPtr(), input.devPtr(), inputSize);
	              cudaDeviceSynchronize();
	              checkError(cudaGetLastError());
	              output.markDeviceDirty();
	              return output;
	              }\n
	            " in
	            generate_functions tail new_str
	          | Ident("reduce") ->
	              let new_str = str ^ head.func_type ^ " " ^
	                    head.kernel_invoke_sym ^
	                        "(VectorArray<" ^ head.func_type ^ "> arr)
	              {
	                  int n = arr.size();
	                  int num_blocks = ceil_div(n, BLOCK_SIZE);
	                  int atob = 1;
	                  int shared_size = BLOCK_SIZE * sizeof(" ^ head.func_type ^ ");
	                  VectorArray<" ^ head.func_type ^ "> tempa(1, num_blocks);
	                  VectorArray<" ^ head.func_type ^ "> tempb(1, num_blocks);

	                  " ^ head.kernel_sym ^ "<<<num_blocks, BLOCK_SIZE, shared_size>>>(tempa.devPtr(), arr.devPtr(), n);
	                  cudaDeviceSynchronize();
	                  checkError(cudaGetLastError());
	                  tempa.markDeviceDirty();
	                  n = num_blocks;

	                  while (n > 1) {
	                      num_blocks = ceil_div(n, BLOCK_SIZE);
	                      if (atob) {
	                          " ^ head.kernel_sym ^ "<<<num_blocks, BLOCK_SIZE, shared_size>>>(tempb.devPtr(), tempa.devPtr(), n);
	                          tempb.markDeviceDirty();
	                      } else {
	                          " ^ head.kernel_sym ^ "<<<num_blocks, BLOCK_SIZE, shared_size>>>(tempa.devPtr(), tempb.devPtr(), n);
	                          tempa.markDeviceDirty();
	                      }
	                      cudaDeviceSynchronize();
	                      checkError(cudaGetLastError());
	                      atob = !atob;
	                      n = num_blocks;
	                  }

	                  if (atob) {
	                      tempa.copyFromDevice(1);
	                      return tempa.elem(false, 0);
	                  }
	                  tempb.copyFromDevice(1);
	                  return tempb.elem(false, 0);
	              }"  in
	              generate_functions tail new_str
	          | _ -> raise Invalid_operation) in
	  generate_functions env.kernel_invocation_functions " "

	let generate_kernel_functions env =
	  let kernel_funcs = env.kernel_functions in
	  let rec generate_funcs funcs str =
	    (match funcs with
	    | [] -> str
	    | head :: tail ->
	        (match head.hof with
	         | Ident("map") ->
	            let new_str = str ^
	            "__global__ void " ^ head.kernel_symbol ^ "(" ^ head.function_type ^ "* output,
	            " ^ head.function_type ^ "* input, size_t n){
	              size_t i = threadIdx.x + blockDim.x * blockIdx.x;
	              if (i < n)
	                output[i] = " ^ head.function_name ^ "(input[i]);
	            }\n"  in
	            generate_funcs tail new_str
	         | Ident("reduce") ->
	             let new_str = str ^
	             "__global__ void " ^ head.kernel_symbol ^ "( " ^ head.function_type ^
	               " *output, " ^ head.function_type ^ " *input, size_t n) {
	                  extern __shared__ " ^ head.function_type ^ " temp[];

	                  int ti = threadIdx.x;
	                  int bi = blockIdx.x;
	                  int starti = blockIdx.x * blockDim.x;
	                  int gi = starti + ti;
	                  int bn = min(n - starti, blockDim.x);
	                  int s;

	                  if (ti < bn)
	                      temp[ti] = input[gi];
	                  __syncthreads();

	                  for (s = 1; s < blockDim.x; s *= 2) {
	                      if (ti % (2 * s) == 0 && ti + s < bn)
	                          temp[ti] = " ^ head.function_name ^ "(temp[ti], temp[ti + s]);
	                      __syncthreads();
	                  }

	                  if (ti == 0)
	                      output[bi] = temp[0];
	              }
	             " in
	             generate_funcs tail new_str

	         | _ -> raise Invalid_operation)) in
	  generate_funcs kernel_funcs ""

	let generate_device_forward_declarations env =
	    let gen_type_list types =
	        if types = [] then "void"
	        else String.concat ", "
	            (List.map
	                (fun typ -> fst (generate_datatype typ env)) types) in
	    let gen_fdecl_if_needed (Ident(id)) (device, rettype, argtypes) genstr =
	        if device then
	            genstr ^ "__device__ " ^ fst (generate_rettype rettype env) ^ " " ^
	            id ^ "(" ^ (gen_type_list argtypes) ^ ");\n"
	        else genstr in
	    FunctionMap.fold gen_fdecl_if_needed env.func_type_map ""

	let generate_pfor_kernels env =
	    let env = Environment.set_on_gpu env in
	    let rec gen_iter_var_decls iter_arr index_var iter_index iterators env =
	        match iterators with
	          | [] -> "", env
	          | ArrayIterator(Ident(id), expr) :: tail -> raise Not_implemented
	          | RangeIterator(id, _) :: tail ->
	                let _, env = Environment.update_scope id Int32 ("", env) in
	                Environment.combine env [
	                    Verbatim("size_t ");
	                    Generator(generate_ident id);
	                    Verbatim(" = get_index_gpu(&" ^ iter_arr ^ "[" ^
	                             string_of_int iter_index ^ "], " ^ index_var ^ ");\n");
	                    Generator(gen_iter_var_decls iter_arr index_var
	                                (iter_index + 1) tail)
	                ] in
	    let rec gen_kernel pfor env =
	        let iter_arr = Symgen.gensym () and
	            niters = Symgen.gensym () and
	            total_iters = Symgen.gensym () and
	            index_var = Symgen.gensym () in
	        Environment.combine env ([
	            Verbatim("__global__ void " ^ pfor.pfor_kernel_name ^ "(");
	            Verbatim("struct range_iter *" ^ iter_arr ^ ", ");
	            Verbatim("size_t " ^ niters ^ ", ");
	            Verbatim("size_t " ^ total_iters)
	        ] @ (if pfor.pfor_arguments = [] then [Verbatim("){\n")] else [
	            Verbatim(", ");
	            Generator(generate_nonempty_decl_list pfor.pfor_arguments);
	            Verbatim("){\n")
	        ]) @ [
	            Verbatim("size_t " ^ index_var ^
	                    " = threadIdx.x + blockIdx.x * blockDim.x;\n");
	            Verbatim("if (" ^ index_var ^ " < " ^ total_iters ^ "){\n");
	            Generator(gen_iter_var_decls iter_arr index_var 0
	                        pfor.pfor_iterators);
	            Generator(generate_statement pfor.pfor_statement);
	            Verbatim("}\n}\n");
	        ]) in
	    let rec generate_kernels str = function
	      | [] -> str
	      | pfor :: tail ->
	            let newstr, _ = Environment.combine env
	                [NewScopeGenerator(gen_kernel pfor)] in
	            generate_kernels (str ^ newstr) tail in
	    generate_kernels "" env.pfor_kernels ;;

	let _ =
	  let lexbuf = Lexing.from_channel stdin in
	  let tree = Parser.top_level Scanner.token lexbuf in
	  let code, env = generate_toplevel tree in
	  let forward_declarations = generate_device_forward_declarations env in
	  let kernel_invocations = generate_kernel_invocation_functions env in
	  let kernel_functions  = generate_kernel_functions env in
	  let pfor_kernels = generate_pfor_kernels env in
	  let header =  "#include <stdio.h>\n\
	                 #include <stdlib.h>\n\
	                 #include <stdint.h>\n\
	                 #include <libvector.hpp>\n\n" in
	  print_string header;
	  print_string forward_declarations;
	  print_string kernel_functions;
	  print_string kernel_invocations;
	  print_string pfor_kernels;
	  print_string code

###A.8 rtlib/libvector.hpp

	#ifndef __LIBVECTOR_H__
	#define __LIBVECTOR_H__

	#include "vector_array.hpp"
	#include "vector_utils.hpp"
	#include "vector_iter.hpp"
	#include <cuComplex.h>
	#endif

###A.9 rtlib/vector_array.hpp

	#ifndef __VECTOR_ARRAY_H__
	#define __VECTOR_ARRAY_H__

	#include <stdlib.h>
	#include <stdarg.h>
	#include "vector_utils.hpp"

	using namespace std;

	struct array_ctrl {
		int refcount;
		char h_dirty;
		char d_dirty;
	};

	template <class T>
	struct device_info {
		T *values;
		size_t *dims;
		size_t ndims;
	};

	template <class T>
	__device__ size_t get_mid_index(struct device_info<T> *info, size_t ind, size_t dim)
	{
		size_t stride = 1;
		size_t i;

		for (i = dim + 1; i < info->ndims; i++)
			stride *= info->dims[i];

		return ind * stride;
	}

	template <class T>
	class VectorArray {
		private:
			T *values;
			T *d_values;
			size_t ndims;
			size_t *dims;
			size_t *d_dims;
			size_t nelems;
			struct array_ctrl *ctrl;
			struct device_info<T> *dev_info;
			size_t bsize();
			void incRef();
			void decRef();
		public:
			VectorArray();
			VectorArray(size_t ndims, ...);
			VectorArray(const VectorArray<T> &orig);
			VectorArray<T> dim_copy(void);
	                T &oned_elem(size_t ind);
			T &elem(bool modify, size_t first_ind, ...);
			VectorArray<T> &chain_set(size_t ind, T val);
			VectorArray<T>& operator= (const VectorArray<T> &orig);
			~VectorArray();
			size_t size();
			size_t length(size_t dim = 0);
			void copyToDevice(size_t n = 0);
			void copyFromDevice(size_t n = 0);
			T *devPtr();
			struct device_info<T> *devInfo();
			void markDeviceDirty(void);
	};

	template <class T>
	VectorArray<T>::VectorArray()
	{
		this->ndims = 0;
		this->dims = NULL;
		this->d_dims = NULL;
		this->values = NULL;
		this->d_values = NULL;
		this->dev_info = NULL;
		this->nelems = 0;
		this->ctrl = (struct array_ctrl *) malloc(sizeof(struct array_ctrl));
		this->ctrl->refcount = 1;
		this->ctrl->h_dirty = 0;
		this->ctrl->d_dirty = 0;
	}

	template <class T>
	VectorArray<T>::VectorArray(size_t ndims, ...)
	{
		size_t i;
		va_list dim_list;
		cudaError_t err;
		struct device_info<T> h_dev_info;

		va_start(dim_list, ndims);

		this->ndims = ndims;
		this->dims = (size_t *) calloc(ndims, sizeof(size_t));
		this->nelems = 1;
		this->ctrl = (struct array_ctrl *) malloc(sizeof(struct array_ctrl));
		this->ctrl->refcount = 1;
		this->ctrl->h_dirty = 0;
		this->ctrl->d_dirty = 0;

		for (i = 0; i < ndims; i++) {
			this->dims[i] = va_arg(dim_list, size_t);
			this->nelems *= this->dims[i];
		}

		va_end(dim_list);

		this->values = (T *) calloc(this->nelems, sizeof(T));
		err = cudaMalloc(&this->d_values, bsize());
		checkError(err);

		err = cudaMalloc(&this->d_dims, sizeof(size_t) * ndims);
		checkError(err);
		err = cudaMemcpy(this->d_dims, this->dims, sizeof(size_t) * ndims,
				cudaMemcpyHostToDevice);
		checkError(err);

		h_dev_info.ndims = ndims;
		h_dev_info.dims = this->d_dims;
		h_dev_info.values = this->d_values;
		err = cudaMalloc(&this->dev_info, sizeof(h_dev_info));
		checkError(err);
		err = cudaMemcpy(this->dev_info, &h_dev_info, sizeof(h_dev_info),
				cudaMemcpyHostToDevice);
		checkError(err);
	}

	template <class T>
	VectorArray<T>::VectorArray(const VectorArray<T> &orig)
	{
		this->ndims = orig.ndims;
		this->dims = orig.dims;
		this->d_dims = orig.d_dims;
		this->nelems = orig.nelems;
		this->ctrl = orig.ctrl;
		this->values = orig.values;
		this->d_values = orig.d_values;
		this->dev_info = orig.dev_info;

		incRef();
	}

	template <class T>
	VectorArray<T> VectorArray<T>::dim_copy(void)
	{
		VectorArray<T> copy;
		cudaError_t err;

		copy.ndims = this->ndims;
		copy.dims = (size_t *) calloc(copy.ndims, sizeof(size_t));

		for (int i = 0; i < this->ndims; i++)
			copy.dims[i] = this->dims[i];

		copy.nelems = this->nelems;
		copy.values = (T *) calloc(this->nelems, sizeof(T));
		err = cudaMalloc(&copy.d_values, copy.bsize());
		checkError(err);

		return copy;
	}

	template <class T>
	VectorArray<T>& VectorArray<T>::operator= (const VectorArray<T>& orig)
	{
		// avoid self-assignment
		if (this == &orig)
			return *this;

		decRef();

		this->ndims = orig.ndims;
		this->dims = orig.dims;
		this->d_dims = orig.d_dims;
		this->nelems = orig.nelems;
		this->ctrl = orig.ctrl;
		this->values = orig.values;
		this->d_values = orig.d_values;
		this->dev_info = orig.dev_info;

		incRef();

		return *this;
	}

	template <class T>
	void VectorArray<T>::incRef(void)
	{
		this->ctrl->refcount++;
	}

	template <class T>
	void VectorArray<T>::decRef(void)
	{
		if (--(this->ctrl->refcount) > 0)
			return;
		free(this->ctrl);
		if (this->dims != NULL)
			free(this->dims);
		if (this->d_dims != NULL)
			cudaFree(this->d_dims);
		if (this->values != NULL)
			free(this->values);
		if (this->d_values != NULL)
			cudaFree(this->d_values);
		if (this->dev_info != NULL)
			cudaFree(this->dev_info);
	}

	template <class T>
	T &VectorArray<T>::oned_elem(size_t ind)
	{
		if (this->ctrl->d_dirty)
			copyFromDevice();

		this->ctrl->h_dirty = 1;
		return this->values[ind];
	}

	template <class T>
	T &VectorArray<T>::elem(bool modify, size_t first_ind, ...)
	{
		size_t ind = first_ind, onedind = first_ind;
		int i;
		va_list indices;

		if (this->ctrl->d_dirty)
			copyFromDevice();

		if (modify)
			this->ctrl->h_dirty = 1;

		va_start(indices, first_ind);

		for (i = 1; i < this->ndims; i++) {
			ind = va_arg(indices, size_t);
			onedind = onedind * this->dims[i] + ind;
		}

		va_end(indices);

		return this->values[onedind];
	}

	template <class T>
	VectorArray<T>::~VectorArray()
	{
		decRef();
	}

	template <class T>
	VectorArray<T>& VectorArray<T>::chain_set(size_t ind, T value)
	{
		oned_elem(ind) = value;
		return *this;
	}

	template <class T>
	size_t VectorArray<T>::size()
	{
		return this->nelems;
	}

	template <class T>
	size_t VectorArray<T>::bsize()
	{
		return sizeof(T) * this->nelems;
	}

	template <class T>
	size_t VectorArray<T>::length(size_t dim)
	{
		return this->dims[dim];
	}

	template <class T>
	void VectorArray<T>::copyToDevice(size_t n)
	{
	        if (n == 0)
	            n = size();
		cudaError_t err;
		err = cudaMemcpy(this->d_values, this->values,
				sizeof(T) * n, cudaMemcpyHostToDevice);
		checkError(err);
		this->ctrl->h_dirty = 0;
	}

	template <class T>
	void VectorArray<T>::copyFromDevice(size_t n)
	{
		cudaError_t err;
	        if (n == 0)
	            n = size();
	        err = cudaMemcpy(this->values, this->d_values,
	                        sizeof(T) * n, cudaMemcpyDeviceToHost);
		checkError(err);
		this->ctrl->d_dirty = 0;
	}

	template <class T>
	T *VectorArray<T>::devPtr()
	{
		if (this->ctrl->h_dirty)
			copyToDevice();
		return this->d_values;
	}

	template <class T>
	struct device_info<T> *VectorArray<T>::devInfo()
	{
		devPtr();
		return this->dev_info;
	}

	template <class T>
	void VectorArray<T>::markDeviceDirty(void)
	{
		this->ctrl->d_dirty = 1;
	}

	#endif

###A.10 rtlib/vector_iter.hpp

	#ifndef __VECTOR_ITER_H__
	#define __VECTOR_ITER_H__

	#include "vector_utils.hpp"

	struct range_iter {
		size_t start;
		size_t stop;
		size_t inc;
		size_t len;
		size_t mod;
		size_t div;
	};

	void fillin_iters(struct range_iter *iters, size_t n)
	{
		int i;
		size_t last_mod = 1;

		for (i = n - 1; i >= 0; i--) {
			iters[i].len = ceil_div(iters[i].stop - iters[i].start,
						iters[i].inc);
			iters[i].div = last_mod;
			iters[i].mod = last_mod * iters[i].len;
			last_mod = iters[i].mod;
		}
	}

	inline size_t get_index_cpu(struct range_iter *iter, size_t oned_ind)
	{
		return iter->start + (oned_ind % iter->mod) / iter->div * iter->inc;
	}

	__device__ size_t get_index_gpu(struct range_iter *iter, size_t oned_ind)
	{
		return iter->start + (oned_ind % iter->mod) / iter->div * iter->inc;
	}

	size_t total_iterations(struct range_iter *iter, size_t n)
	{
		int total = 1;
		size_t i;

		for (i = 0; i < n; i++)
			total *= iter[i].len;

		return total;
	}

	struct range_iter *device_iter(struct range_iter *iters, size_t n)
	{
		cudaError_t err;
		struct range_iter *d_iters;

		err = cudaMalloc(&d_iters, n * sizeof(struct range_iter));
		checkError(err);
		err = cudaMemcpy(d_iters, iters, n * sizeof(struct range_iter),
				cudaMemcpyHostToDevice);
		checkError(err);

		return d_iters;
	}

	#endif

###A.11 rtlib/vector_utils.hpp

	#ifndef __VECTOR_UTILS_H__
	#define __VECTOR_UTILS_H__

	#include <sys/time.h>

	static inline void _check(cudaError_t err, const char *file, int line)
	{
	    if (err != cudaSuccess) {
	        fprintf(stderr, "CUDA error at %s:%d\n", file, line);
	        fprintf(stderr, "%s\n", cudaGetErrorString(err));
	        exit(err);
	    }
	}

	static inline double get_time(void)
	{
		struct timeval tv;

		gettimeofday(&tv, NULL);

		return (double) tv.tv_sec + ((double) tv.tv_usec) / 1000000.0;
	}

	#ifndef BLOCK_SIZE
	#define BLOCK_SIZE 1024
	#endif
	#define checkError(err) _check((err), __FILE__, __LINE__)
	#define ceil_div(n, d) (((n) - 1) / (d) + 1)
	#define min(a, b) (((a) < (b)) ? (a) : (b))
	#define max(a, b) (((a) > (b)) ? (a) : (b))

	#endif

