%{ open Ast %}

%token LPAREN RPAREN EQUAL
%token PLUS MINUS TIMES DIVIDE
%token SEMICOLON LCURLY RCURLY
%token EOF
%token <int> INT_LITERAL
%token <string> IDENTIFIER
%token <string> PRIMITIVE_TYPE

%left SEMICOLON
%left EQUAL
%left PLUS MINUS
%left TIMES DIVIDE

%start expr
%type <Ast.expr> expr

%%

expr:
    expr PLUS expr   { Binop($1, Add, $3) }
  | expr MINUS expr  { Binop($1, Sub, $3) }
  | expr TIMES expr  { Binop($1, Mul, $3) }
  | expr DIVIDE expr { Binop($1, Div, $3) }
  | IDENTIFIER EQUAL expr { Assign($1, $3) }
  | LPAREN expr RPAREN { $2 }
  | INT_LITERAL        { IntLit($1) }
  | IDENTIFIER         { Ident($1) }

statement:
    LCURLY RCURLY {}
  | LCURLY statement_seq RCURLY {}
  | expr SEMICOLON {}
  | SEMICOLON {}

statement_seq:
    statement_seq statement {}
  | statement {}

%%
