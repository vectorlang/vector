open Environment
open Ast

exception Not_allowed_on_gpu

let combine_detect_tuples tuplist =
    let combine_pair tup1 tup2 =
        let (in1, out1) = tup1 and
            (in2, out2) = tup2 in
        (in1 @ in2, out1 @ out2) in
    List.fold_left combine_pair ([], []) tuplist;;

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
      | _ -> ([], [])
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
                    ([], [])
                else if ins then
                    ([ident], [])
                else ([], [])
            else ([], [])
      | ArrayElem(expr, indices) ->
            let (indins, indouts) as indtups =
                detect_expr_list indices env in
            (match expr with
              | Lval(Variable(ident)) ->
                    if var_in_scope ident env then
                        ((if ins then ident :: indins else indins),
                         (if outs then ident :: indouts else indouts))
                    else indtups
              | _ -> combine_detect_tuples [detect_expr expr env; indtups])
      | ComplexAccess(expr, _) ->
            (match expr with
              | Lval(Variable(ident)) ->
                    if var_in_scope ident env then
                        ((if ins then [ident] else []),
                         (if outs then [ident] else []))
                    else ([], [])
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

