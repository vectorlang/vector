open Ast

(* The environment is a stack of (VariableMap * FunctionMap) tuples. When a new
 * scope is entered, we push to the stack; when a scope is left, we pop. *)

(* Maps variable idents to type *)
module VariableMap = Map.Make(struct type t = ident let compare = compare end);;

(* Maps function idents to return type *)
module FunctionMap = Map.Make(struct type t = ident let compare = compare end);;

module FunctionDeclarationMap = Map.Make(struct type t = ident let compare = compare end);;

exception Already_declared
exception Invalid_environment
exception Invalid_operation

type 'a sourcecomponent =
  | Verbatim of string
  | Generator of ('a -> (string * 'a))
  | NewScopeGenerator of ('a -> (string * 'a))
let create =
  ([], FunctionDeclarationMap.empty, FunctionMap.empty, VariableMap.empty :: []);;

let get_var_type ident env =
  let rec check_scope scopes =
    match scopes with
     | [] -> raise Not_found
     | scope :: tail ->
         if VariableMap.mem ident scope then
           VariableMap.find ident scope
         else
           check_scope tail in
  let _, _, _, scope_stack = env in
  check_scope scope_stack

let is_var_declared ident env =
  let _, _,_, scope_stack = env in
  match scope_stack with
   | [] -> false
   | scope :: tail -> VariableMap.mem ident scope

let set_var_type ident datatype env =
  let global_funcs, func_content_map, func_map, scope_stack = env in
  let scope, tail = (match scope_stack with
                | scope :: tail -> scope, tail
                | [] -> raise Invalid_environment) in
  let new_scope = VariableMap.add ident datatype scope in
  global_funcs, func_content_map, func_map, new_scope :: tail

let update_scope ident datatype (str, env) =
  if is_var_declared ident env then
    raise Already_declared
  else
    (str, set_var_type ident datatype env)

let push_scope (global_funcs, func_content_map, func_map, scope_stack) =
  global_funcs, func_content_map, func_map, VariableMap.empty :: scope_stack

let pop_scope (global_funcs, func_content_map, func_map, scope_stack) =
  match scope_stack with
   | local_scope :: tail -> global_funcs, func_content_map, func_map, tail
   | [] -> raise Invalid_environment

let get_func_type ident (_,_,func_map, _) =
  FunctionMap.find ident func_map

let is_func_declared ident env =
  let _,_, func_map, _ = env in
  FunctionMap.mem ident func_map

let update_global_funcs hof (str, env) =
  let global_funcs, func_content_map, func_map, scope_stack = env in
  let new_global_funcs = hof :: global_funcs in
  (str, (new_global_funcs, func_content_map, func_map, scope_stack))

let set_func_type ident returntype env =
  let global_funcs, func_content_map, func_map, scope_stack = env in
  let new_func_map = FunctionMap.add ident returntype func_map in
  global_funcs, func_content_map, new_func_map, scope_stack

let update_function_content function_decl (str, env) = 
  let global_funcs, func_content_map, func_map, scope_stack = env in
  let new_func_content_map = match function_decl with 
    | FunctionDecl(t, ident, ds, ss) -> FunctionDeclarationMap.add ident (FunctionDecl(t,ident,ds,ss)) func_content_map
    | _ -> raise Invalid_operation in
  (str, (global_funcs, new_func_content_map, func_map, scope_stack))

let update_functions ident returntype (str, env) =
  if is_func_declared ident env then
    raise Already_declared
  else
    (str, set_func_type ident returntype env)

let rec render_global_functions env =
  let global_funcs, func_content, func_map, scope_stack = env in
  match global_funcs with
  | [] -> print_endline("")
  | HigherOrderFunctionCall(hof, ident, expr) :: tail -> (match hof with
      | Ident("map") ->
          "__global__ void " ^ "function_placeholder" ^ "map(#{type} *result, #{type} *input, size_t n) {

              size_t i = threadidx.x + blockdim.x * blockidx.x;
             
              #{varname} = input[i]
              
                #{content} 
              
              result[i] = #{return value} 
            }
          "   
      | Ident("reduce") -> raise Not_implemented
      | _ -> raise Invalid_operation)
  | _  :: tail -> raise Invalid_operation

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
