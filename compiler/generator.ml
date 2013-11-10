open Ast
open Complex
open Environment
open Symgen

module StringMap = Map.Make(String);;

exception Unknown_type
exception Empty_list
exception Type_mismatch
exception Not_implemented (* this should go away *)
exception Invalid_operation

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
      | ComplexLit(re, im) ->
          (match (infer_type re env), (infer_type im env) with
            | (Float, Float) -> Complex
            | (Float32, Float32) -> Complex64
            | (Float64, Float64) -> Complex128
            | _ -> raise Type_mismatch)
      | FloatLit(_) -> Float64
      | Int64Lit(_) -> Int64
      | IntLit(_) -> Int
      | StringLit(_) -> ArrayType(Char)
      | ArrayLit(exprs) ->
          let f expr = infer_type expr env in
          ArrayType(match_type (List.map f exprs))
      | Cast(datatype, expr) -> datatype
      | Lval(lval) -> (match lval with
          | ArrayElem(e, _) -> infer_type e env
          | Variable(i) -> Environment.get_var_type i env
          | ComplexAccess(expr1,ident) ->
              (match (infer_type expr1 env) with
                | Complex -> Float
                | Complex64 -> Float32
                | Complex128 -> Float64
                | _ -> raise Invalid_operation))
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

      | HigherOrderFunctionCall(hof, f, expr) ->
          (match(hof) with
            | Ident("map") -> infer_type expr env
            | Ident("reduce") -> raise Not_implemented
            | _ -> raise Invalid_operation)


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
     | Double -> Environment.combine env [Verbatim("double")]
     | Float -> Environment.combine env [Verbatim("float")]
     | Float32 -> Environment.combine env [Verbatim("float")]
     | Float64 -> Environment.combine env [Verbatim("double")]


     | Complex -> Environment.combine env [Verbatim("cuFloatComplex")]
     | Complex64 -> Environment.combine env [Verbatim("cuFloatComplex")]
     | Complex128 -> Environment.combine env [Verbatim("cuDoubleComplex")]


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
        | Complex | Complex64 | Complex128 ->
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
      let len_unop e = Environment.combine env [
          Verbatim("(");
          Generator(generate_expr e);
          Verbatim(").size()")
      ] in
      match op with
          Neg -> simple_unop "-" e
        | LogNot -> simple_unop "!" e
        | BitNot -> simple_unop "~" e
        | Len -> len_unop e
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
  | ComplexLit(re, im) as lit ->
      let make_func = (match (infer_type lit env) with
        | Complex | Complex64 -> "make_cuFloatComplex"
        | Complex128 -> "make_cuDoubleComplex"
        | _ -> raise Type_mismatch) in
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
      (match infer_type es env with
        | ArrayType(function_type) ->
            let function_type_str, _ = generate_datatype function_type env in
            let kernel_invoke_sym = Symgen.gensym () in
            let kernel_sym = Symgen.gensym () in
            let function_name, _ = generate_ident i2 env in
            Environment.update_global_funcs function_type_str kernel_invoke_sym function_name i1 kernel_sym (Environment.combine env [
                Verbatim(kernel_invoke_sym ^ "(");
                Generator(generate_expr es);
                Verbatim(");")
            ])
        | _ -> raise Type_mismatch)
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
           NewScopeGenerator(generate_for_statement (is, s));
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
         let new_func_sym = Symgen.gensym () in
         let str, env = Environment.combine env [
             NewScopeGenerator(generate_function (t,i,ds,ss))
           ] in
         let mod_str, _ = Environment.combine env [
           NewScopeGenerator(generate_function (t,Ident (new_func_sym),ds,ss))
         ] in
         let new_str, new_env = Environment.update_functions i t (str, env) in
         let final_str, final_env = Environment.update_function_content new_str mod_str new_func_sym i new_env in
         final_str, final_env

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
      let output_sym = match iterator with
        ArrayIterator(i,_) -> i
      | RangeIterator(i,_) -> i
      in
      (* start_sym and inc_sym are never actually used for array iterators...
       * the only consequence is that the generated symbols in our output code
       * will be non-consecutive because of these "wasted" identifiers *)
      let start_sym = Ident(Symgen.gensym () ^ iter_name iterator ^ "_start") in
      let inc_sym = Ident(Symgen.gensym () ^ iter_name iterator ^ "_inc") in
      (iterator, len_sym, mod_sym, output_sym, start_sym, inc_sym)
    in

    List.fold_left (fun m i -> StringMap.add (iter_name i) (get_iter_properties i) (m)) (StringMap.empty) (iterators)
  in

  (* generate code to calculate the length of each iterator
   * and initialize the corresponding variables
   *
   * also calculate start and inc *)
  let iter_length_initializers =

    let iter_length_inits _ (iter, len_sym, _, _, start_sym, inc_sym) acc =
      match iter with
        ArrayIterator(_,e) -> [ Declaration(AssigningDecl(len_sym, Unop(Len, e))) ] :: acc
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
    StringMap.fold (fun _ (_, len_sym, _, _, _, _) acc ->
      Binop(acc, Mul, Lval(Variable(len_sym)))) (iter_map) (IntLit(Int32.of_int 1))
  in

  (* figure out how often each iterator wraps around
   * the rightmost iterator wraps the fastest (i.e. mod its own length)
   * all other iterators wrap modulo (their own length times the mod of
   * the iterator directly to the right) *)
  let iter_mod_initializers =
    let iter_initializer iterator acc =
      let name = iter_name iterator in
      let mod_ident, len_ident = match (StringMap.find name iter_map) with (_,l,m,_,_,_) -> m,l in
      match acc with
        [] -> [ Declaration(AssigningDecl(mod_ident, Lval(Variable(len_ident)))) ]
      | Declaration(AssigningDecl(prev_mod_ident, _)) :: _ -> (
          Declaration(AssigningDecl(mod_ident, Binop(Lval(Variable(len_ident)), Mul, Lval(Variable(prev_mod_ident)))))
          :: acc)
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

  (* the variables that our iterators are actually generating *)
  (* TODO: proper type inference for the output of array iterators *)
  let output_initializers =
    let iter_ident iterator = match iterator with
      ArrayIterator(i,_) -> i
    | RangeIterator(i,_) -> i
    in
    List.map (fun iter -> Declaration(PrimitiveDecl(Int, iter_ident iter))) (iterators)
  in

  (* these assignments will occur at the beginning of each iteration *)
  let output_assignments =
    let iter_properties s = match (StringMap.find s iter_map) with
      (_, _, mod_sym, output_sym, start_sym, inc_sym) ->
        (mod_sym, output_sym, start_sym, inc_sym)
    in
    let idx mod_sym = Binop(Lval(Variable(iter_ptr_ident)), Div, Lval(Variable(mod_sym))) in
    let iter_assignment iterator = match iterator with
      (* TODO: to avoid unnecessary copies, we really want to use pointers here *)
      ArrayIterator(Ident(s),e) -> (
        let mod_sym, output_sym, _, _ = iter_properties s in
        (* TODO: we really should store the result of e in a variable, to
         * avoid evaluating it more than once *)
        Expression(Assign(Variable(output_sym), Lval(ArrayElem(e, [idx mod_sym]))))
      )
    | RangeIterator(Ident(s),_) -> (
        let mod_sym, output_sym, start_sym, inc_sym = iter_properties s in
        let offset = Binop(idx mod_sym, Mul, Lval(Variable(inc_sym))) in
        let origin = Lval(Variable(start_sym)) in
        Expression(Assign(Variable(output_sym), Binop(origin, Add, offset)))
      )
    in
    List.map (iter_assignment) (iterators)
  in

  Environment.combine env [
    Verbatim("{\n");
    Generator(generate_statement_list iter_length_initializers);
    Generator(generate_statement_list iter_mod_initializers);
    Generator(generate_statement_list bounds_initializers);
    Generator(generate_statement_list output_initializers);
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

let generate_toplevel tree =
    let env = Environment.create in
    Environment.combine env [
        Generator(generate_statement_list tree);
        Verbatim("\nint main(void) { return vec_main(); }\n")
    ]

let generate_kernel_invocation_functions env =
  let kernel_funcs, _, _, _, _ = env in
  let rec generate_functions funcs str =
    match funcs with
    | [] -> str
    | head :: tail ->
        (let kernel_invoke_sym, kernel_sym, function_type  = head in
        let new_str = str ^ "\nVectorArray<" ^ function_type ^ "> " ^ kernel_invoke_sym ^ "(" ^
        "VectorArray<" ^ function_type ^ "> input){
          int inputSize = input.size();
          VectorArray<" ^ function_type ^ " > output = array_init<" ^ function_type ^ ">((size_t) inputSize);
          input.copyToDevice();
          " ^ kernel_sym ^
          "<<<ceil_div(inputSize,BLOCK_SIZE),BLOCK_SIZE>>>(output.devPtr(), input.devPtr(), inputSize);
          cudaDeviceSynchronize();
          checkError(cudaGetLastError());
          output.copyFromDevice();
          return output;
          }\n
        " in
        generate_functions tail new_str) in
  generate_functions kernel_funcs " "

let generate_kernel_functions env =
  let kernel_funcs, global_funcs, func_content, func_map, scope_stack = env in
  let rec generate_funcs funcs str =
    (match funcs with
    | [] -> str
    | head :: tail ->
        let function_name, hof, kernel_sym, function_type = head in
        let symbolized_function_name, _ = Environment.lookup_function_content function_name func_content in
        (match hof with
         | Ident("map") ->
            let new_str = str ^
            "__global__ void " ^ kernel_sym ^ "(" ^ function_type ^ "* output,
            " ^ function_type ^ "* input, size_t n){
              size_t i = threadIdx.x + blockDim.x * blockIdx.x;
              if (i < n)
                output[i] = " ^ symbolized_function_name ^ "(input[i]);
            }\n"  in
            generate_funcs tail new_str
         | Ident("reduce") -> raise Not_implemented
         | _ -> raise Invalid_operation)) in
  generate_funcs global_funcs ""

let generate_device_functions env =
  let kernel_funcs, global_funcs, func_content_map, func_map, scope_stack = env in
  let rec generate_funcs funcs str =
    match funcs with
    | [] -> str
    | head :: tail ->
        let function_name, hof, kernel_sym, function_type = head in
        let _, func_content = Environment.lookup_function_content function_name func_content_map in
        let new_str = str ^ "\n__device__ " ^ func_content ^ "\n" in
        generate_funcs tail new_str in

  generate_funcs global_funcs ""

let _ =
  let lexbuf = Lexing.from_channel stdin in
  let tree = Parser.top_level Scanner.token lexbuf in
  let code, env = generate_toplevel tree in
  let device_functions =  generate_device_functions env in
  let kernel_invocations = generate_kernel_invocation_functions env in
  let kernel_functions  = generate_kernel_functions env in
  let header =  "#include <stdio.h>\n\
                  #include <stdlib.h>\n\
                  #include <stdint.h>\n\
                  #include <libvector.hpp>\n\n" in
  print_string header;
  print_string device_functions;
  print_string kernel_functions;
  print_string kernel_invocations;
  print_string code
