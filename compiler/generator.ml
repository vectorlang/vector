open Ast
open Complex

exception Unknown_type
exception Empty_list
exception Type_mismatch
exception Not_implemented (* this should go away *)

let rec infer_type expr =
    let f type1 type2 =
        match type1 with
          | Some(t) -> (if t = type2 then Some(t) else raise Type_mismatch)
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
         | _ -> match_type [infer_type expr1; infer_type expr2])
      | CharLit(_) -> Char
      | ComplexLit(_) -> raise Not_implemented
      | FloatLit(_) -> Float
      | Int64Lit(_) -> Int64
      | IntLit(_) -> Int
      | StringLit(_) -> ArrayType(Char)
      | ArrayLit(exprs) -> ArrayType(match_type (List.map infer_type exprs))
      | Cast(datatype, expr) -> datatype
      | Lval(lval) -> (match lval with
          | ArrayElem(e, _) -> infer_type e
          | Variable(_) -> raise Not_implemented)
      | AssignOp(lval, _, expr) ->
            let l = Lval(lval) in
            match_type [infer_type l; infer_type expr]
      | Unop(op, expr) -> (if op = Neg then Bool else infer_type expr)
      | PostOp(lval, _) -> let l = Lval(lval) in infer_type l
      | Assign(lval, expr) ->
            let l = Lval(lval) in
            match_type [infer_type l; infer_type expr]
      | FunctionCall(i, _) -> raise Not_implemented
      | HigherOrderFunctionCall(hof, f, expr_list) -> raise Not_implemented (*
      probably something like the result of f *)

let generate_ident = function
    Ident(s) -> s

let generate_datatype s =
    match s with
     | Bool -> "bool"
     | Char -> "char"
     | Int8 -> "int8_t"
     | UInt8 -> "uint8_t"
     | Int16 -> "int16_t"
     | UInt16 -> "uint16_t"
     | Int -> "int"
     | Int32 -> "int32_t"
     | UInt -> "uint"
     | UInt32 -> "uint32_t"
     | Int64 -> "int64_t"
     | UInt64 -> "uint64_t"
     | _ -> raise Unknown_type

let rec generate_lvalue = function
    Variable(i) -> generate_ident i
  | ArrayElem(e, es) -> generate_expr e ^ ".elem(" ^
        generate_expr_list es ^ ")"
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
  | Unop(op,e) -> (
      let _op = match op with
          Neg -> "-"
        | LogNot -> "!"
        | BitNot -> "~"
      in
      _op ^ generate_expr e
    )
  | PostOp(lvalue, op) -> (
      let _op = match op with
          Dec -> "--"
        | Inc -> "++"
      in
      generate_lvalue lvalue ^ _op
    )
  | Assign(lvalue, e) -> generate_lvalue lvalue ^ " = " ^ generate_expr e
  | IntLit(i) -> Int32.to_string i
  | Int64Lit(i) -> Int64.to_string i
  | FloatLit(f) -> string_of_float f
  | ComplexLit(c) -> "(" ^ string_of_float c.re ^ " + i" ^ string_of_float c.im ^ ")"
  | StringLit(s) -> "\"" ^ s ^ "\""
  | CharLit(c) -> "'" ^ Char.escaped c ^ "'"
  | ArrayLit(es) -> "array_init(" ^ generate_expr_list es ^ ")"
  | Cast(d,e) -> "(" ^ generate_datatype d ^ ") (" ^ generate_expr e ^ ")"
  | FunctionCall(i,es) -> generate_ident i ^ "(" ^ generate_expr_list es ^ ")"
  | HigherOrderFunctionCall(i1,i2,es) -> "@" ^ generate_ident i1 ^ "(" ^ generate_ident i2 ^ ", " ^ generate_expr_list es ^ ")"
  | Lval(lvalue) -> generate_lvalue lvalue
and generate_expr_list = function
    [] -> ""
  | hd :: tl as lst -> generate_nonempty_expr_list lst
and generate_nonempty_expr_list = function
    expr :: [] -> generate_expr expr
  | expr :: tl -> generate_expr expr ^ ", " ^ generate_nonempty_expr_list tl
  | [] -> raise Empty_list
and generate_decl = function
    AssigningDecl(i,e) ->
        let t = (infer_type e) in
        (match t with
          | ArrayType(f) -> raise Not_implemented
          | _ -> generate_datatype t ^ " " ^ generate_ident i ^ " = " ^ generate_expr e)
  | PrimitiveDecl(d,i) -> generate_datatype d ^ " " ^ generate_ident i
  | ArrayDecl(d,i,es) ->
        let arrinit = (match es with
            [] -> ""
          | _ -> "(" ^ string_of_int (List.length es) ^
                ", " ^ generate_expr_list es ^ ")") in
        "VectorArray<" ^ generate_datatype d ^ "> " ^
            generate_ident i ^ arrinit

let generate_range = function
    Range(e1,e2,e3) -> "range(" ^ generate_expr e1 ^ ", " ^ generate_expr e2 ^ ", " ^ generate_expr e3 ^ ")"

let generate_iterator = function
    RangeIterator(i, r) -> generate_ident i ^ " in " ^ generate_range r
  | ArrayIterator(i, e) -> generate_ident i ^ " in " ^ generate_expr e

let rec generate_iterator_list = function
    [] -> "_"
  | hd :: tl -> generate_iterator hd ^ ", " ^ generate_iterator_list tl

let rec generate_nonempty_decl_list = function
    hd :: [] -> generate_decl hd
  | hd :: tl -> generate_decl hd ^ ", " ^ generate_nonempty_decl_list tl
  | [] -> raise Empty_list

let generate_decl_list = function
    [] -> "void"
  | hd :: tl as lst -> generate_nonempty_decl_list lst

let rec generate_statement = function
    CompoundStatement(ss) -> "{\n" ^ generate_statement_list ss ^ "}\n"
  | Declaration(d) -> generate_decl d ^ ";"
  | Expression(e) -> generate_expr e ^ ";"
  | IncludeStatement(s) -> "include \"" ^ s ^ "\";"
  | EmptyStatement -> ";"
  | IfStatement(e, s1, s2) -> "if (" ^ generate_expr e ^ ")\n" ^
        generate_statement s1 ^ "\nelse\n" ^ generate_statement s2
  | WhileStatement(e, s) -> "while (" ^ generate_expr e ^ ")\n" ^
        generate_statement s
  | ForStatement(is, s) -> "for (" ^ generate_iterator_list is ^ ") {\n" ^ generate_statement s ^ "}"
  | PforStatement(is, s) -> "pfor (" ^ generate_iterator_list is ^ ") {\n" ^ generate_statement s ^ "}"
  | FunctionDecl(t, i, ds, ss) -> generate_datatype t ^ " " ^ generate_ident i ^ "(" ^ generate_decl_list ds ^ ") {\n" ^ generate_statement_list ss ^ "}"
  | ForwardDecl(t, i, ds) -> generate_datatype t ^ " " ^ generate_ident i ^ "(" ^ generate_decl_list ds ^ ");"
  | ReturnStatement(e) -> "return " ^ generate_expr e ^ ";"
  | VoidReturnStatement -> "return;"
  | SyncStatement -> "sync;"

and generate_statement_list = function
    [] -> ""
  | hd :: tl -> generate_statement hd ^ "\n" ^ generate_statement_list tl

let generate_toplevel tree =
    "#include <stdio.h>\n" ^
    "#include <stdlib.h>\n" ^
    "#include <stdint.h>\n" ^
    "#include <libvector.hpp>\n\n" ^
    generate_statement_list tree ^
    "\nint main(void) { return vec_main(); }"

let _ =
  let lexbuf = Lexing.from_channel stdin in
  let tree = Parser.top_level Scanner.token lexbuf in
  let code = generate_toplevel tree in
  print_string code
