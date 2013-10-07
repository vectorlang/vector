{ open Parser }

let decdigit = ['0'-'9']
let hexdigit = ['0'-'9' 'a'-'f' 'A'-'Z']

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

        | decdigit+ | "0x" hexdigit+
            as lit { INT_LITERAL(Int32.of_string lit) }
        | (decdigit+ | "0x" hexdigit+ as lit) 'L'
            { INT64_LITERAL(Int64.of_string lit) }
        | decdigit+ '.' decdigit* | '.' decdigit+
        | decdigit+ ('.' decdigit*)? 'e' '-'? decdigit+
            as lit { FLOAT_LITERAL(float_of_string lit) }
        | '"' (('\\' _ | [^ '"'])* as str) '"'
            { STRING_LITERAL(Scanf.unescaped(str)) }
        | '\'' ('\\' _ | [^ '\''] | "\\x" hexdigit hexdigit as lit) '\''
            { CHAR_LITERAL((Scanf.unescaped(lit)).[0]) }

        | "bool" | "char" | "byte" | "int" | "uint"
        | "int8" | "uint8" | "int16" | "uint16"
        | "int32" | "uint32" | "int64" | "uint64"
        | "float" | "float32" | "double" | "float64"
        | "complex" | "complex64" | "complex128"
            as primtype { TYPE(primtype) }

        | ['a'-'z' 'A'-'Z' '_']+ as ident { IDENT(ident) }

        | eof { EOF }
