%{ open Ast %}

%token LPAREN RPAREN LCURLY RCURLY LSQUARE RSQUARE
%token DOT COMMA SC
%token EQUAL DECL_EQUAL
%token LSHIFT RSHIFT BITAND BITOR BITXOR LOGAND LOGOR
%token LT LTE GT GTE EE NE
%token PLUS MINUS TIMES DIVIDE MODULO
%token LOGNOT BITNOT DEC INC
%token EOF
%token <int32> INT_LITERAL
%token <int64> INT64_LITERAL
%token <float> FLOAT_LITERAL
%token <string> IDENT TYPE STRING_LITERAL
%token <char> CHAR_LITERAL

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
%nonassoc UMINUS LOGNOT BITNOT DEC INC
%nonassoc LPAREN RPAREN LSQUARE RSQUARE

%start statement_seq
%type <Ast.expr> expr
%type <Ast.statement> statement
%type <Ast.statement list> statement_seq

%%

expr:
    expr PLUS expr   { Binop($1, Add, $3) }
  | expr MINUS expr  { Binop($1, Sub, $3) }
  | expr TIMES expr  { Binop($1, Mul, $3) }
  | expr DIVIDE expr { Binop($1, Div, $3) }
  | expr MODULO expr { Binop($1, Mod, $3) }
  | expr LSHIFT expr { Binop($1, Lshift, $3) }
  | expr RSHIFT expr { Binop($1, Rshift, $3) }
  | expr LT expr     { Binop($1, Less, $3) }
  | expr LTE expr    { Binop($1, LessEq, $3) }
  | expr GT expr     { Binop($1, Greater, $3) }
  | expr GTE expr    { Binop($1, GreaterEq, $3) }
  | expr EE expr     { Binop($1, Eq, $3) }
  | expr NE expr     { Binop($1, NotEq, $3) }
  | expr BITAND expr { Binop($1, BitAnd, $3) }
  | expr BITXOR expr { Binop($1, BitXor, $3) }
  | expr BITOR expr  { Binop($1, BitOr, $3) }
  | expr LOGAND expr { Binop($1, LogAnd, $3) }
  | expr LOGOR expr  { Binop($1, LogOr, $3) }

  | MINUS expr %prec UMINUS { Preop(Neg, $2) }
  | LOGNOT expr { Preop(LogNot, $2) }
  | BITNOT expr { Preop(BitNot, $2) }
  | DEC expr    { Preop(PreDec, $2) }
  | INC expr    { Preop(PreInc, $2) }

  | expr DEC { Postop($1, PostDec) }
  | expr INC { Postop($1, PostInc) }

  | LPAREN expr RPAREN { $2 }

  | IDENT EQUAL expr { Assign($1, $3) }
  | IDENT            { Ident($1) }

  | INT_LITERAL    { IntLit($1) }
  | INT64_LITERAL  { Int64Lit($1) }
  | FLOAT_LITERAL  { FloatLit($1) }
  | STRING_LITERAL { StringLit($1) }
  | CHAR_LITERAL   { CharLit($1) }

statement:
    LCURLY RCURLY { CompoundStatement([]) }
  | LCURLY statement_seq RCURLY { CompoundStatement($2) }
  | expr SC { Expression($1) }
  | IDENT DECL_EQUAL expr SC { AssigningDecl($1, $3) }
  | TYPE IDENT SC { PrimitiveDecl($1, $2) }
  | TYPE IDENT LSQUARE RSQUARE SC { ArrayDecl($1, $2, IntLit(0l)) }
  | TYPE IDENT LSQUARE expr RSQUARE SC { ArrayDecl($1, $2, $4) }
  | SC { EmptyStatement }

statement_seq:
    statement statement_seq { $1 :: $2 }
  | statement { $1 :: [] }

%%
