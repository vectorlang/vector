(* The environment is a stack of (VariableMap * FunctionMap) tuples. When a new
 * scope is entered, we push to the stack; when a scope is left, we pop. *)

(* Maps variable idents to type *)
module VariableMap = Map.Make(String);;

(* Maps function idents to return type *)
module FunctionMap = Map.Make(String);;

type 'a sourcecomponent =
  | Verbatim of string
  | Generator of ('a -> (string * 'a))

let create =
  (FunctionMap.empty, VariableMap.empty :: []);;

let update initial_env components =
    let f (str, env) component =
        match component with
         | Verbatim(verbatim) -> str ^ verbatim, env
         | Generator(gen) ->
           let new_str, new_env = gen env in
            str ^ new_str, new_env in
    List.fold_left f ("", initial_env) components
