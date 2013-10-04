%{ open Ast %}

%token SEMICOLON LPAREN RPAREN EQUAL
%token PLUS MINUS TIMES DIVIDE
%token EOF
%token <int> INT_LITERAL
%token <string> IDENTIFIER

%left SEMICOLON
%left EQUAL
%left PLUS MINUS
%left TIMES DIVIDE

%start expr
%type <Ast.expr> expr

%%

expr:
    expr SEMICOLON expr  { Seq($1, $3) }
  | expr EQUAL expr  { Assign($1, $3) }
  | expr PLUS expr   { Binop($1, Add, $3) }
  | expr MINUS expr  { Binop($1, Sub, $3) }
  | expr TIMES expr  { Binop($1, Mul, $3) }
  | expr DIVIDE expr { Binop($1, Div, $3) }
  | LPAREN expr RPAREN { $2 }
  | INT_LITERAL        { IntLit($1) }
  | IDENTIFIER         { Ident($1) }

%%
