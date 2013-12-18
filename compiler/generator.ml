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
exception Syntax_error of int * int * string

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
            | Ident("map") -> ArrayType(let (_,dtype,_) =
                  Environment.get_func_info f env in dtype)
            | Ident("reduce") -> let (_,dtype,_) =
                  Environment.get_func_info f env in dtype
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
         | Ident(sym) -> raise (Symbol_not_found sym) in
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
          let env = if device then set_on_gpu env else env in
          let str, env = Environment.combine env [
              NewScopeGenerator(generate_function device return_type
                                identifier arg_list body_sequence)
            ] in
          let env = if device then clear_on_gpu env else env in
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

(* TODO: support multi-dimensional arrays *)
(* TODO: clean up this humongous mess *)
(* TODO: support negative integers in ranges *)
(* TODO: modify the type system so we can use size_t where appropriate *)
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

    List.fold_left (fun m i -> StringMap.add (iter_name i)
                    (get_iter_properties i) (m)) (StringMap.empty) (iterators)
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
          (* TODO: make sure we never get negative lengths *)
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
      let (_, len_ident, mod_ident, div_ident, _, _, _) =
          StringMap.find name iter_map in
      match acc with
        [] -> [
          Declaration(AssigningDecl(mod_ident, Lval(Variable(len_ident))));
          Declaration(AssigningDecl(div_ident, IntLit(Int32.of_int 1)));
        ]
      | Declaration(AssigningDecl(prev_mod_ident, _)) :: _ -> (
          List.append [
            Declaration(AssigningDecl(mod_ident, Binop(
                Lval(Variable(len_ident)), Mul,
                Lval(Variable(prev_mod_ident)))));
            Declaration(AssigningDecl(div_ident, Lval(Variable(prev_mod_ident))));
          ] acc)
      | _ -> [] (* TODO: how do we represent impossible outcomes? *)
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
      (* TODO: to avoid unnecessary copies, we really want to use pointers here *)
      ArrayIterator(Ident(s), Lval(Variable(ident))) -> (
        let mod_sym, div_sym, output_sym, _, _ = iter_properties s in
        (* TODO: we really should store the result of e in a variable, to
         * avoid evaluating it more than once *)
        Declaration(AssigningDecl(output_sym,
            Lval(ArrayElem(ident,[idx mod_sym div_sym])))))
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
    Generator(generate_expr (Binop(Lval(Variable(iter_ptr_ident)), Less,
        Lval(Variable(iter_max_ident)))));
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
            let new_str = str ^ "\nVectorArray<" ^ head.func_type ^ "> " ^
                head.kernel_invoke_sym ^ "(" ^
            "VectorArray<" ^ head.func_type ^ "> input){
              int inputSize = input.size();
              VectorArray<" ^ head.func_type ^ " > output = input.dim_copy();
              " ^ head.kernel_sym ^
              "<<<ceil_div(inputSize,BLOCK_SIZE),BLOCK_SIZE>>>" ^
              "(output.devPtr(), input.devPtr(), inputSize);
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

                  " ^ head.kernel_sym ^
                  "<<<num_blocks, BLOCK_SIZE, shared_size>>>" ^
                  "(tempa.devPtr(), arr.devPtr(), n);
                  cudaDeviceSynchronize();
                  checkError(cudaGetLastError());
                  tempa.markDeviceDirty();
                  n = num_blocks;

                  while (n > 1) {
                      num_blocks = ceil_div(n, BLOCK_SIZE);
                      if (atob) {
                          " ^ head.kernel_sym ^
                          "<<<num_blocks, BLOCK_SIZE, shared_size>>>" ^
                          "(tempb.devPtr(), tempa.devPtr(), n);
                          tempb.markDeviceDirty();
                      } else {
                          " ^ head.kernel_sym ^
                          "<<<num_blocks, BLOCK_SIZE, shared_size>>>" ^
                          "(tempa.devPtr(), tempb.devPtr(), n);
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
  let env = set_on_gpu env in
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
    let tree = try
        Parser.top_level Scanner.token lexbuf
    with except ->
        let curr = lexbuf.Lexing.lex_curr_p in
        let line = curr.Lexing.pos_lnum in
        let col = curr.Lexing.pos_cnum in
        let tok = Lexing.lexeme lexbuf in
        raise (Syntax_error (line, col, tok))
    in
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
