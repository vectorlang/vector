open Ast

let generate_ident = function
    Ident(s) -> "$" ^ s

let rec generate_lvalue = function
    Variable(i) -> generate_ident i
  | ArrayElem(e, es) -> "()"
and generate_expr = function
    Binop(e1,op,e2) -> (
      let _op = match op with
          Add -> "+"
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
      generate_expr e1 ^ " " ^ _op ^ " " ^ generate_expr e2
    )
  | AssignOp(lvalue, op, e) -> (
      let _op = match op with
          AddAssn -> "+="
        | SubAssn -> "-="
        | MulAssn -> "*="
        | DivAssn -> "/="
        | ModAssn -> "%="
        | LshiftAssn -> "<<="
        | RshiftAssn -> ">>="
        | BitOrAssn -> "|="
        | BitAndAssn -> "&="
        | BitXorAssn -> "^="
      in
      generate_lvalue lvalue ^ " " ^ _op ^ " " ^ generate_expr e
    )
  | Assign(lvalue, e) -> generate_lvalue lvalue ^ " = " ^ generate_expr e
  | IntLit(i) -> Int32.to_string i
  | Int64Lit(i) -> Int64.to_string i
  | StringLit(s) -> "\"" ^ s ^ "\""
  | Lval(lvalue) -> generate_lvalue lvalue
  | _ -> "_"

let generate_datatype = function
  Type(s) -> "#" ^ s

let generate_decl = function
    AssigningDecl(i,e) -> generate_ident i ^ " := " ^ generate_expr e
  | PrimitiveDecl(d,i) -> generate_datatype d ^ " " ^ generate_ident i
  | _ -> "()"

let rec generate_statement = function
    CompoundStatement(ss) -> generate_statement_list ss
  | Declaration(d) -> generate_decl d ^ ";"
  | Expression(e) -> generate_expr e ^ ";"
  | IncludeStatement(s) -> "include \"" ^ s ^ "\";"
  | EmptyStatement -> "_"
  | IfStatement(e, s1, s2) -> "if (" ^ generate_expr e ^ ") {\n" ^ generate_statement s1 ^ "} else {\n" ^ generate_statement s2 ^ "}"
  | FunctionDecl(t, i, ds, ss) ->
    let rec generate_decl_list = function
        [] -> ""
      | hd :: tl -> generate_decl hd ^ ", " ^ generate_decl_list tl
    in
    generate_datatype t ^ " " ^ generate_ident i ^ "(" ^ generate_decl_list ds ^ ") {\n" ^ generate_statement_list ss ^ "}"
  | ReturnStatement(e) -> "return " ^ generate_expr e ^ ";"
  | VoidReturnStatement -> "return;"
  | SyncStatement -> "sync;"
  | _ -> "_"

and generate_statement_list = function
    [] -> ""
  | hd :: tl -> generate_statement hd ^ "\n" ^ generate_statement_list tl

let _ =
  let lexbuf = Lexing.from_channel stdin in
  let tree = Parser.top_level Scanner.token lexbuf in
  let code = generate_statement_list tree in
  print_string code
