{ open Parser }

let decdigit = ['0'-'9']
let hexdigit = ['0'-'9' 'a'-'f' 'A'-'Z']
let floating = decdigit+ '.' decdigit* | '.' decdigit+
        | decdigit+ ('.' decdigit*)? 'e' '-'? decdigit+

rule token =
    parse [' ' '\t' '\r' '\n'] { token lexbuf }
        | ';' { SC }
        | '.' { DOT }
        | ',' { COMMA }
        | '(' { LPAREN }
        | ')' { RPAREN }
        | '{' { LCURLY }
        | '}' { RCURLY }
        | '[' { LSQUARE }
        | ']' { RSQUARE }
        | '=' { EQUAL }
        | ":=" { DECL_EQUAL }

        | '+' { PLUS }
        | '-' { MINUS }
        | '*' { TIMES }
        | '/' { DIVIDE }
        | '%' { MODULO }
        | "<<" { LSHIFT }
        | ">>" { RSHIFT }
        | '<' { LT }
        | "<=" { LTE }
        | '>' { GT }
        | ">=" { GTE }
        | "==" { EE }
        | "!=" { NE }
        | '&' { BITAND }
        | '^' { BITXOR }
        | '|' { BITOR }
        | "&&" { LOGAND }
        | "||" { LOGOR }

        | '!' { LOGNOT }
        | '~' { BITNOT }
        | "++" { INC }
        | "--" { DEC }

        | "+=" { PLUS_EQUALS }
        | "-=" { MINUS_EQUALS }
        | "*=" { TIMES_EQUALS }
        | "/-" { DIVIDE_EQUALS }
        | "%=" { MODULO_EQUALS }
        | "<<+" { LSHIFT_EQUALS }
        | ">>=" { RSHIFT_EQUALS }
        | "|=" { BITOR_EQUALS }
        | "&=" { BITAND_EQUALS }
        | "^=" { BITXOR_EQUALS }

        | decdigit+ | "0x" hexdigit+
            as lit { INT_LITERAL(Int32.of_string lit) }
        | (decdigit+ | "0x" hexdigit+ as lit) 'L'
            { INT64_LITERAL(Int64.of_string lit) }
        | floating as lit { FLOAT_LITERAL(float_of_string lit) }
        | "#(" (floating as real) ',' ' '? (floating as imag) ')'
            { COMPLEX_LITERAL(
                {Complex.re = float_of_string real;
                 Complex.im = float_of_string imag}) }
        | '"' (('\\' _ | [^ '"'])* as str) '"'
            { STRING_LITERAL(Scanf.unescaped(str)) }
        | '\'' ('\\' _ | [^ '\''] | "\\x" hexdigit hexdigit as lit) '\''
            { CHAR_LITERAL((Scanf.unescaped(lit)).[0]) }

        | "bool" | "char" | "byte" | "int" | "uint"
        | "int8" | "uint8" | "int16" | "uint16"
        | "int32" | "uint32" | "int64" | "uint64"
        | "float" | "float32" | "double" | "float64"
        | "complex" | "complex64" | "complex128"
        | "void"
            as primtype { TYPE(primtype) }

        | "if" {IF}
        | "else" {ELSE}
        | ['a'-'z' 'A'-'Z' '_']+ decdigit* as ident { IDENT(ident) }

        | "/*" { comments lexbuf }
        | "//" {inline_comments lexbuf}

        | eof { EOF }
and comments = parse
  | "*/"                { token lexbuf}
  | _                   { comments lexbuf}
and inline_comments = parse
  | "\n"  {token lexbuf}
  | _ {inline_comments lexbuf}
