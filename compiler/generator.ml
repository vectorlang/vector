open Ast

let generate = function
  _ -> print_endline "hello"

let _ =
  let lexbuf = Lexing.from_channel stdin in
  let tree = Parser.top_level Scanner.token lexbuf in
  generate tree
