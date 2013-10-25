open Ast

let generate_ident = function
    Ident(s) -> "$" ^ s

let rec generate_expr = function
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
      "(" ^ generate_expr e1 ^ " " ^ _op ^ " " ^ generate_expr e2 ^ ")"
    )
  | IntLit(i) -> Int32.to_string i
  | Int64Lit(i) -> Int64.to_string i
  | StringLit(s) -> "\"" ^ s ^ "\""
  | _ -> "()"

let generate_decl = function
    AssigningDecl(i,e) -> "(decl (" ^ generate_ident i ^ " := " ^ generate_expr e ^ "))"
  | _ -> "()"

let rec generate_statement = function
    Declaration(d) -> generate_decl d
  | Expression(e) -> generate_expr e
  | IncludeStatement(s) -> "(include \"" ^ s ^ "\")"
  | ReturnStatement(e) -> "(return " ^ generate_expr e ^ ")"
  | VoidReturnStatement -> "(return)"
  | SyncStatement -> "(sync)"
  | _ -> "()"

let rec generate_toplevel = function
    [] -> ""
  | hd :: tl -> generate_statement hd ^ "\n" ^ generate_toplevel tl

let _ =
  let lexbuf = Lexing.from_channel stdin in
  let tree = Parser.top_level Scanner.token lexbuf in
  let code = generate_toplevel tree in
  print_string code
