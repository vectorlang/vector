{ open Parser }

rule token =
    parse [' ' '\t' '\r' '\n'] { token lexbuf }
        | ';' { SEMICOLON }
        | '(' { LPAREN }
        | ')' { RPAREN }
        | '{' { LCURLY }
        | '}' { RCURLY }
        | '=' { EQUAL }
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

        | ['0'-'9']+ | "0x" ['0'-'9' 'a'-'f' 'A'-'F']
            as lit { INT_LITERAL(int_of_string lit) }
        | "bool" | "char" | "byte" | "int" | "uint"
        | "int8" | "uint8" | "int16" | "uint16"
        | "int32" | "uint32" | "int64" | "uint64"
        | "float" | "float32" | "double" | "float64"
        | "complex" | "complex64" | "complex128"
            as primtype { PRIMITIVE_TYPE(primtype) }
        | ['a'-'z' 'A'-'Z' '_']+ as ident { IDENTIFIER(ident) }
        | eof { EOF }
