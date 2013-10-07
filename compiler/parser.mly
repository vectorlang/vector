%{ open Ast %}

%token LPAREN RPAREN LCURLY RCURLY LSQUARE RSQUARE
%token DOT COMMA SC
%token EQUAL DECL_EQUAL
%token LSHIFT RSHIFT BITAND BITOR BITXOR LOGAND LOGOR
%token LT LTE GT GTE EE NE
%token PLUS MINUS TIMES DIVIDE MODULO
%token EOF
%token <int> INT_LITERAL
%token <string> IDENT TYPE

%left SC
%left DECL_EQUAL EQUAL
%left LOGOR
%left LOGAND
%left BITOR
%left BITXOR
%left BITAND
%left EE NE
%left LT LTE GT GTE
%left LSHIFT RSHIFT
%left PLUS MINUS
%left TIMES DIVIDE MODULO
%nonassoc LPAREN RPAREN LSQUARE RSQUARE

%start statement_seq
%type <Ast.expr> expr
%type <unit> statement
%type <unit> statement_seq

%%

expr:
    expr PLUS expr   { Binop($1, Add, $3) }
  | expr MINUS expr { Binop($1, Sub, $3) }
  | expr TIMES expr { Binop($1, Mul, $3) }
  | expr DIVIDE expr { Binop($1, Div, $3) }
  | expr MODULO expr { Binop($1, Mod, $3) }
  | expr LSHIFT expr { Binop($1, Lshift, $3) }
  | expr RSHIFT expr { Binop($1, Rshift, $3) }
  | expr LT expr { Binop($1, Less, $3) }
  | expr LTE expr { Binop($1, LessEq, $3) }
  | expr GT expr { Binop($1, Greater, $3) }
  | expr GTE expr { Binop($1, GreaterEq, $3) }
  | expr EE expr { Binop($1, Eq, $3) }
  | expr NE expr { Binop($1, NotEq, $3) }
  | expr BITAND expr { Binop($1, BitAnd, $3) }
  | expr BITXOR expr { Binop($1, BitXor, $3) }
  | expr BITOR expr { Binop($1, BitOr, $3) }
  | expr LOGAND expr { Binop($1, LogAnd, $3) }
  | expr LOGOR expr { Binop($1, LogOr, $3) }

  | IDENT EQUAL expr { Assign($1, $3) }
  | LPAREN expr RPAREN { $2 }
  | INT_LITERAL        { IntLit($1) }
  | IDENT         { Ident($1) }

statement:
    LCURLY RCURLY {}
  | LCURLY statement_seq RCURLY {}
  | expr SEMICOLON {}
  | expr DECL_EQUAL SEMICOLON {}
  | PRIMITIVE_TYPE IDENTIFIER SEMICOLON {}
  | PRIMITIVE_TYPE IDENTIFIER EQUAL expr SEMICOLON {}
  | SEMICOLON {}

statement_seq:
    statement statement_seq {}
  | statement {}

%%
