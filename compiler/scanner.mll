{ open Parser }

rule token =
    parse [' ' '\t' '\r' '\n'] { token lexbuf }
        | ';' { SEMICOLON }
        | '(' { LPAREN }
        | ')' { RPAREN }
        | '=' { EQUAL }
        | '+' { PLUS }
        | '-' { MINUS }
        | '*' { TIMES }
        | '/' { DIVIDE }
        | ['0'-'9']+ | "0x" ['0'-'9' 'a'-'f' 'A'-'F']
            as lit { INT_LITERAL(int_of_string lit) }
        | ['a'-'z' 'A'-'Z' '_']+ as ident { IDENTIFIER(ident) }
        | eof { EOF }
