open Ast

let generate = function
  _ -> "hello"

let _ =
  let lexbuf = Lexing.from_channel stdin in
  let tree = Parser.top_level Scanner.token lexbuf in
  let code = generate tree in
  print_endline code
