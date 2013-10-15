%{ open Ast %}

%token LPAREN RPAREN LCURLY RCURLY LSQUARE RSQUARE
%token DOT COMMA SC COLON
%token EQUAL DECL_EQUAL
%token PLUS_EQUALS MINUS_EQUALS TIMES_EQUALS DIVIDE_EQUALS MODULO_EQUALS
%token LSHIFT_EQUALS RSHIFT_EQUALS BITOR_EQUALS BITAND_EQUALS BITXOR_EQUALS
%token LSHIFT RSHIFT BITAND BITOR BITXOR LOGAND LOGOR
%token LT LTE GT GTE EE NE
%token PLUS MINUS TIMES DIVIDE MODULO
%token LOGNOT BITNOT DEC INC
%token IF ELSE WHILE FOR PFOR IN
%token RETURN VOID SYNC
%token EOF
%token <int32> INT_LITERAL
%token <int64> INT64_LITERAL
%token <float> FLOAT_LITERAL
%token <string> IDENT TYPE STRING_LITERAL
%token <char> CHAR_LITERAL
%token <Complex.t> COMPLEX_LITERAL

%left SC
%right DECL_EQUAL EQUAL PLUS_EQUALS MINUS_EQUALS TIMES_EQUALS DIVIDE_EQUALS MODULO_EQUALS LSHIFT_EQUALS RSHIFT_EQUALS BITOR_EQUALS BITAND_EQUALS BITXOR_EQUALS
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
%right UMINUS LOGNOT BITNOT DEC INC
%nonassoc LPAREN RPAREN LSQUARE RSQUARE
%nonassoc IFX
%nonassoc ELSE

%start top_level
%type <Ast.statement list> top_level

%%

ident:
    IDENT { Ident($1) }

datatype:
    TYPE { Type($1) }

lvalue:
  | ident { Variable($1) }
  | expr LSQUARE expr_list RSQUARE { ArrayElem($1, $3) }

expr:
  | expr PLUS expr   { Binop($1, Add, $3) }
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

  | lvalue PLUS_EQUALS expr   { AssignOp($1, AddAssn, $3) }
  | lvalue MINUS_EQUALS expr  { AssignOp($1, SubAssn, $3) }
  | lvalue TIMES_EQUALS expr  { AssignOp($1, MulAssn, $3) }
  | lvalue DIVIDE_EQUALS expr { AssignOp($1, DivAssn, $3) }
  | lvalue MODULO_EQUALS expr { AssignOp($1, ModAssn, $3) }
  | lvalue LSHIFT_EQUALS expr { AssignOp($1, LshiftAssn, $3) }
  | lvalue RSHIFT_EQUALS expr { AssignOp($1, RshiftAssn, $3) }
  | lvalue BITOR_EQUALS expr  { AssignOp($1, BitOrAssn, $3) }
  | lvalue BITAND_EQUALS expr { AssignOp($1, BitAndAssn, $3) }
  | lvalue BITXOR_EQUALS expr { AssignOp($1, BitXorAssn, $3) }

  | MINUS expr %prec UMINUS { Preop(Neg, $2) }
  | LOGNOT expr { Preop(LogNot, $2) }
  | BITNOT expr { Preop(BitNot, $2) }
  | DEC expr    { Preop(PreDec, $2) }
  | INC expr    { Preop(PreInc, $2) }

  | expr DEC { Postop($1, PostDec) }
  | expr INC { Postop($1, PostInc) }

  | LPAREN expr RPAREN { $2 }

  | lvalue EQUAL expr { Assign($1, $3) }
  | lvalue            { Lval($1) }

  | INT_LITERAL                 { IntLit($1) }
  | INT64_LITERAL               { Int64Lit($1) }
  | FLOAT_LITERAL               { FloatLit($1) }
  | COMPLEX_LITERAL             { ComplexLit($1) }
  | STRING_LITERAL              { StringLit($1) }
  | CHAR_LITERAL                { CharLit($1) }
  | datatype LPAREN expr RPAREN { Cast($1, $3) }
  | LCURLY expr_list RCURLY     { ArrayLit($2) }

  | ident LPAREN RPAREN               { FunctionCall($1, []) }
  | ident LPAREN expr_list RPAREN { FunctionCall ($1, $3) }

expr_list:
  | expr COMMA expr_list { $1 :: $3 }
  | expr                 { [$1] }

decl:
  | ident DECL_EQUAL expr SC               { AssigningDecl($1, $3) }
  | datatype ident SC                      { PrimitiveDecl($1, $2) }
  | datatype ident LSQUARE RSQUARE SC      { ArrayDecl($1, $2, []) }
  | datatype ident LSQUARE expr_list RSQUARE SC { ArrayDecl($1, $2, $4) }

statement:
  | IF LPAREN expr RPAREN statement ELSE statement
      { IfStatement($3, $5, $7) }
  | IF LPAREN expr RPAREN statement %prec IFX
      { IfStatement($3, $5, EmptyStatement) }

  | WHILE LPAREN expr RPAREN statement { WhileStatement($3, $5) }
  | FOR LPAREN iterator_list RPAREN statement { ForStatement($3, $5) }
  | PFOR LPAREN iterator_list RPAREN statement { PforStatement($3, $5) }

  | LCURLY statement_seq RCURLY { CompoundStatement($2) }

  | expr SC { Expression($1) }
  | SC { EmptyStatement }
  | decl { Declaration($1) }

  | RETURN expr SC { ReturnStatement($2) }
  | RETURN SC { VoidReturnStatement }

  | SYNC SC { SyncStatement }

iterator_list:
  | iterator COMMA iterator_list { $1 :: $3 }
  | iterator { [$1] }

iterator:
  | ident IN range { RangeIterator($1, $3) }
  | ident IN expr { ArrayIterator($1, $3) }

range:
  | expr COLON expr COLON expr { Range($1, $3, $5) }
  | expr COLON expr { Range($1, $3, IntLit(1l)) }
  | COLON expr COLON expr { Range(IntLit(0l), $2, $4) }
  | COLON expr { Range(IntLit(0l), $2, IntLit(1l)) }

top_level_statement:
  | datatype ident LPAREN param_list RPAREN LCURLY statement_seq RCURLY
      { FunctionDecl($1, $2, $4, $7) }
  | decl { Declaration($1) }

param:
  | datatype ident { PrimitiveDecl($1, $2) }
  | datatype ident LSQUARE RSQUARE
      { ArrayDecl($1, $2, []) }

non_empty_param_list:
  | param COMMA non_empty_param_list { $1 :: $3 }
  | param { [$1] }

param_list:
  | non_empty_param_list { $1 }
  | { [] }

top_level:
  | top_level_statement top_level {$1 :: $2}
  | top_level_statement { [$1] }

statement_seq:
  | statement statement_seq {$1 :: $2 }
  | { [] }

%%
