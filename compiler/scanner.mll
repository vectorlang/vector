{ open Parser;; exception Illegal_identifier }

let decdigit = ['0'-'9']
let hexdigit = ['0'-'9' 'a'-'f' 'A'-'Z']
let letter = ['a'-'z' 'A'-'Z' '_']
let floating =
    decdigit+ '.' decdigit* | '.' decdigit+
  | decdigit+ ('.' decdigit*)? 'e' '-'? decdigit+
  | '.' decdigit+ 'e' '-'? decdigit+

rule token = parse
  | [' ' '\t' '\r' '\n'] { token lexbuf }
  | ';' { SC }
  | ':' { COLON }
  | '.' { DOT }
  | ',' { COMMA }
  | '@' { AT }
  | '(' { LPAREN }  | ')' { RPAREN }
  | '{' { LCURLY }  | '}' { RCURLY }
  | '[' { LSQUARE } | ']' { RSQUARE }
  | '=' { EQUAL } | ":=" { DECL_EQUAL }

  | '+' { PLUS } | '-' { MINUS } | '*' { TIMES } | '/' { DIVIDE }
  | '%' { MODULO }
  | "<<" { LSHIFT } | ">>" { RSHIFT }
  | '<' { LT } | "<=" { LTE }
  | '>' { GT } | ">=" { GTE }
  | "==" { EE } | "!=" { NE }
  | '&' { BITAND } | '^' { BITXOR } | '|' { BITOR }
  | "&&" { LOGAND } | "||" { LOGOR }

  | '!' { LOGNOT } | '~' { BITNOT }
  | "++" { INC } | "--" { DEC }
  | "len" { LEN }

  | "+=" { PLUS_EQUALS } | "-=" { MINUS_EQUALS }
  | "*=" { TIMES_EQUALS } | "/-" { DIVIDE_EQUALS }
  | "%=" { MODULO_EQUALS }
  | "<<+" { LSHIFT_EQUALS } | ">>=" { RSHIFT_EQUALS }
  | "|=" { BITOR_EQUALS } | "&=" { BITAND_EQUALS } | "^=" { BITXOR_EQUALS }

  | '#' { HASH }

  | decdigit+ | "0x" hexdigit+
      as lit { INT_LITERAL(Int32.of_string lit) }
  | (decdigit+ | "0x" hexdigit+ as lit) 'L'
      { INT64_LITERAL(Int64.of_string lit) }
  | floating as lit { FLOAT_LITERAL(float_of_string lit) }
  | '"' (('\\' _ | [^ '"'])* as str) '"'
      { STRING_LITERAL(str) }
  | '\'' ('\\' _ | [^ '\''] | "\\x" hexdigit hexdigit as lit) '\''
      { CHAR_LITERAL((Scanf.unescaped(lit)).[0]) }

  | "include" {INCLUDE}

  | "bool" | "char" | "byte" | "int" | "uint"
  | "int8" | "uint8" | "int16" | "uint16"
  | "int32" | "uint32" | "int64" | "uint64"
  | "float" | "float32" | "double" | "float64"
  | "complex" | "complex64" | "complex128"
  | "void" | "string"
      as primtype { TYPE(primtype) }

  | "return" { RETURN }
  | "sync" { SYNC }
  | "if" { IF }
  | "else" { ELSE }
  | "for" { FOR }
  | "while" { WHILE }
  | "pfor" { PFOR }
  | "in" { IN }

  | "__sym" decdigit+ "_" letter* { raise Illegal_identifier }
  | letter (letter | decdigit)* as ident { IDENT(ident) }

  | "/*" { comments lexbuf }
  | "//" {inline_comments lexbuf}

  | eof { EOF }

and comments = parse
  | "*/"                { token lexbuf}
  | _                   { comments lexbuf}

and inline_comments = parse
  | "\n"  {token lexbuf}
  | _ {inline_comments lexbuf}
