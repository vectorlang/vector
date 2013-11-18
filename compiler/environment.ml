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

type 'a env = {
  kernel_invocation_functions: kernel_invocation_function list;
  kernel_functions: kernel_function list;
  func_decl_map: 'a FunctionDeclarationMap.t;
  func_type_map: datatype FunctionMap.t;
  scope_stack: datatype VariableMap.t list;
}
type 'a sourcecomponent =
  | Verbatim of string
  | Generator of ('a -> (string * 'a))
  | NewScopeGenerator of ('a -> (string * 'a))

let create =
  {
    kernel_invocation_functions = [];
    kernel_functions = [];
    func_decl_map = FunctionDeclarationMap.empty;
    func_type_map = FunctionMap.empty;
    scope_stack = VariableMap.empty ::[];
  }

let update_env kernel_invocation_functions kernel_functions func_decl_map
  func_type_map var_map_list =
    {
      kernel_invocation_functions = kernel_invocation_functions;
      kernel_functions = kernel_functions;
      func_decl_map = func_decl_map;
      func_type_map = func_type_map;
      scope_stack = var_map_list;
    }
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
    env.func_decl_map env.func_type_map (new_scope :: tail)

let update_scope ident datatype (str, env) =
  if is_var_declared ident env then
    raise Already_declared
  else
    (str, set_var_type ident datatype env)

let push_scope env =
  update_env env.kernel_invocation_functions env.kernel_functions
    env.func_decl_map env.func_type_map (VariableMap.empty :: env.scope_stack)

let pop_scope env =
  match env.scope_stack with
   | local_scope :: tail ->
      update_env env.kernel_invocation_functions env.kernel_functions
        env.func_decl_map env.func_type_map tail
   | [] -> raise Invalid_environment

let get_func_type ident env =
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

  (str, update_env new_kernel_funcs new_global_funcs env.func_decl_map
  env.func_type_map env.scope_stack)

let set_func_type ident returntype env =
  let new_func_type_map = FunctionMap.add ident returntype env.func_type_map in
  update_env env.kernel_invocation_functions env.kernel_functions
    env.func_decl_map new_func_type_map env.scope_stack

let update_function_content main_str new_func_body new_func_sym ident env =
  let new_func_content_map = FunctionDeclarationMap.add ident (new_func_sym,
  new_func_body) env.func_decl_map in
  (main_str, update_env env.kernel_invocation_functions env.kernel_functions
    new_func_content_map env.func_type_map env.scope_stack)

let update_functions ident returntype (str, env) =
  if is_func_declared ident env then
    raise Already_declared
  else
    (str, set_func_type ident returntype env)

let lookup_function_content function_name function_content_map =
  FunctionDeclarationMap.find (Ident (function_name)) function_content_map

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