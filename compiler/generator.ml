open Ast
open Complex
open Environment

exception Unknown_type
exception Empty_list
exception Type_mismatch
exception Not_implemented (* this should go away *)


let rec infer_type expr env =
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
         | _ -> match_type [infer_type expr1 env; infer_type expr2 env])
      | CharLit(_) -> Char
      | ComplexLit(_) -> Complex
      | FloatLit(_) -> Float
      | Int64Lit(_) -> Int64
      | IntLit(_) -> Int
      | StringLit(_) -> ArrayType(Char)
      | ArrayLit(exprs) ->
          let f expr = infer_type expr env in
          ArrayType(match_type (List.map f exprs))
      | Cast(datatype, expr) -> datatype
      | Lval(lval) -> (match lval with
          | ArrayElem(e, _) -> infer_type e env
          | Variable(i) -> Environment.get_var_type i env)
      | AssignOp(lval, _, expr) ->
            let l = Lval(lval) in
            match_type [infer_type l env; infer_type expr env]
      | Unop(op, expr) -> (if op = Neg then Bool else infer_type expr env)
      | PostOp(lval, _) -> let l = Lval(lval) in infer_type l env
      | Assign(lval, expr) ->
            let l = Lval(lval) in
            match_type [infer_type l env; infer_type expr env]
      | FunctionCall(i, _) -> Environment.get_func_type i env
      (* this depends on the HOF type: ex map is int list -> int list *)
      | HigherOrderFunctionCall(hof, f, expr_list) -> raise Not_implemented


let generate_ident ident env =
  match ident with
    Ident(s) -> Environment.combine env [Verbatim(s)]

let generate_datatype datatype env =
    match datatype with
     | Bool -> Environment.combine env [Verbatim("bool")]
     | Char -> Environment.combine env [Verbatim("char")]
     | Int8 -> Environment.combine env [Verbatim("int8_t")]
     | UInt8 -> Environment.combine env [Verbatim("uint8_t")]
     | Int16 -> Environment.combine env [Verbatim("int16_t")]
     | UInt16 -> Environment.combine env [Verbatim("uint16_t")]
     | Int -> Environment.combine env [Verbatim("int")]
     | Int32 -> Environment.combine env [Verbatim("int32_t")]
     | UInt -> Environment.combine env [Verbatim("uint")]
     | UInt32 -> Environment.combine env [Verbatim("uint32_t")]
     | Int64 -> Environment.combine env [Verbatim("int64_t")]
     | UInt64 -> Environment.combine env [Verbatim("uint64_t")]
     | _ -> raise Unknown_type

let rec generate_lvalue lval env =
  match lval with
   | Variable(i) ->
       Environment.combine env [Generator(generate_ident i)]
   | ArrayElem(e, es) ->
       Environment.combine env [
         Generator(generate_expr e);
         Verbatim(".elem(");
         Generator(generate_expr_list es);
         Verbatim(")")
       ]
and generate_expr expr env =
  match expr with
    Binop(e1,op,e2) ->
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
        Generator(generate_expr e1);
        Verbatim(" " ^ _op ^ " ");
        Generator(generate_expr e2)
      ]
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
      Environment.combine env [
        Generator(generate_lvalue lvalue);
        Verbatim(" " ^ _op ^ " ");
        Generator(generate_expr e)
      ]
    )
  | Unop(op,e) -> (
      let _op = match op with
          Neg -> "-"
        | LogNot -> "!"
        | BitNot -> "~"
      in
      Environment.combine env [
        Verbatim(_op);
        Generator(generate_expr e)
      ]
    )
  | PostOp(lvalue, op) -> (
      let _op = match op with
          Dec -> "--"
        | Inc -> "++"
      in
      Environment.combine env [
        Generator(generate_lvalue lvalue);
        Verbatim(_op)
      ]
    )
  | Assign(lvalue, e) ->
      Environment.combine env [
        Generator(generate_lvalue lvalue);
        Verbatim(" = ");
        Generator(generate_expr e)
      ]
  | IntLit(i) ->
      Environment.combine env [Verbatim(Int32.to_string i)]
  | Int64Lit(i) ->
      Environment.combine env [Verbatim(Int64.to_string i)]
  | FloatLit(f) ->
      Environment.combine env [Verbatim(string_of_float f)]
  | ComplexLit(c) ->
      Environment.combine env [
        Verbatim("(" ^ string_of_float c.re);
        Verbatim(" + i" ^ string_of_float c.im ^ ")")
      ]
  | StringLit(s) ->
      Environment.combine env [Verbatim("\"" ^ s ^ "\"")]
  | CharLit(c) ->
      Environment.combine env [
        Verbatim("'" ^ Char.escaped c ^ "'")
      ]
  | ArrayLit(es) ->
      let typ = (match (infer_type (ArrayLit(es)) env) with
       | ArrayType(t) -> t
       | _ -> raise Type_mismatch) in
      let len = Int32.of_int (List.length es) in
      Environment.combine env [
        Verbatim("array_init<");
        Generator(generate_datatype typ);
        Verbatim(">((size_t) ");
        Generator(generate_expr_list (IntLit(len) :: es));
        Verbatim(")")
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
      Environment.combine env [
        Generator(generate_ident i);
        Verbatim("(");
        Generator(generate_expr_list es);
        Verbatim(")")
      ]
  | HigherOrderFunctionCall(i1,i2,es) ->
      Environment.combine env [
        Verbatim("@");
        Generator(generate_ident i1);
        Verbatim("(");
        Generator(generate_ident i2);
        Verbatim(", ");
        Generator(generate_expr_list es);
        Verbatim(")")
      ]
  | Lval(lvalue) ->
      Environment.combine env [Generator(generate_lvalue lvalue)]
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
       Environment.update_scope ident datatype (match datatype with
          | ArrayType(f) ->
              Environment.combine env [
                Verbatim("VectorArray<");
                Generator(generate_datatype f);
                Verbatim("> ");
                Generator(generate_ident ident);
                Verbatim(" = ");
                Generator(generate_expr e);
              ]
          | _ ->
              Environment.combine env [
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
       Environment.update_scope i d (match es with
          | [] -> "", env
          | _ ->
              Environment.combine env [
                Verbatim("VectorArray<");
                Generator(generate_datatype d);
                Verbatim("> ");
                Generator(generate_ident i);
                Verbatim("(" ^ string_of_int (List.length es) ^ ", ");
                Generator(generate_expr_list es);
                Verbatim(")")
              ])

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
     | IncludeStatement(s) ->
         Environment.combine env [
           Verbatim("include \"" ^ s ^ "\";")
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
           Verbatim("for (");
           NewScopeGenerator(generate_iterator_list is);
           Verbatim(") {\n");
           NewScopeGenerator(generate_statement s);
           Verbatim("}")
         ]
     | PforStatement(is, s) ->
         Environment.combine env [
           Verbatim("pfor (");
           NewScopeGenerator(generate_iterator_list is);
           Verbatim(") {\n");
           NewScopeGenerator(generate_statement s);
           Verbatim("}")
         ]
     | FunctionDecl(t, i, ds, ss) ->
         Environment.update_functions i t (Environment.combine env [
             NewScopeGenerator(generate_function (t,i,ds,ss))
           ])
     | ForwardDecl(t, i, ds) ->
         Environment.update_functions i t (Environment.combine env [
           Generator(generate_datatype t);
           Verbatim(" ");
           Generator(generate_ident i);
           Verbatim("(");
           Generator(generate_decl_list ds);
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
     | SyncStatement ->
         Environment.combine env [Verbatim("sync;")]

and generate_statement_list statement_list env =
    match statement_list with
     | [] -> Environment.combine env []
     | statement :: tail ->
         Environment.combine env [
           Generator(generate_statement statement);
           Verbatim("\n");
           Generator(generate_statement_list tail)
         ]

and generate_function (returntype, ident, params, statements) env =
         Environment.combine env [
           Generator(generate_datatype returntype);
           Verbatim(" ");
           Generator(generate_ident ident);
           Verbatim("(");
           Generator(generate_decl_list params);
           Verbatim(") {\n");
           Generator(generate_statement_list statements);
           Verbatim("}");
         ]

let generate_toplevel tree =
    let env = Environment.create in
    Environment.combine env [
        Verbatim("#include <stdio.h>\n\
                  #include <stdlib.h>\n\
                  #include <stdint.h>\n\
                  #include <libvector.hpp>\n\n");
        Generator(generate_statement_list tree);
        Verbatim("\nint main(void) { return vec_main(); }")
    ]

let _ =
  let lexbuf = Lexing.from_channel stdin in
  let tree = Parser.top_level Scanner.token lexbuf in
  let code, _ = generate_toplevel tree in
  print_string code
