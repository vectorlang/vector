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
      | EmptyStatement -> (IdentSet.empty, IdentSet.empty)
      | _ -> raise Not_allowed_on_gpu
and detect_decl decl env =
    match decl with
      | AssigningDecl(ident, expr) -> detect_expr expr env
      | _ -> (IdentSet.empty, IdentSet.empty)
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
