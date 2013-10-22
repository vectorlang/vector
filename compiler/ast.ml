type binop = Add | Sub | Mul | Div | Mod
    | Lshift | Rshift
    | Less | LessEq | Greater | GreaterEq | Eq | NotEq
    | BitAnd | BitXor | BitOr | LogAnd | LogOr

type assignop =
      AddAssn | SubAssn | MulAssn | DivAssn | ModAssn
    | LshiftAssn | RshiftAssn
    | BitOrAssn | BitAndAssn | BitXorAssn ;;

type preop = Neg | LogNot | BitNot | PreDec | PreInc

type postop = PostDec | PostInc

type datatype =
    Type of string

type ident =
    Ident of string

type lvalue =
  | Variable of ident
  | ArrayElem of expr * expr list
and expr =
    Binop of expr * binop * expr
  | AssignOp of lvalue * assignop * expr
  | Preop of preop * expr
  | Postop of expr * postop
  | Assign of lvalue * expr
  | IntLit of int32
  | Int64Lit of int64
  | FloatLit of float
  | ComplexLit of Complex.t
  | StringLit of string
  | CharLit of char
  | ArrayLit of expr list
  | Cast of datatype * expr
  | FunctionCall of ident * expr list
  | HigherOrderFunctionCall of ident * ident * expr list
  | Lval of lvalue

type decl =
    AssigningDecl of ident * expr
  | PrimitiveDecl of datatype * ident
  | ArrayDecl of datatype * ident * expr list

type range =
    Range of expr * expr * expr

type iterator =
    RangeIterator of ident * range
  | ArrayIterator of ident * expr

type statement =
    CompoundStatement of statement list
  | Declaration of decl
  | Expression of expr
  | IncludeStatement of string
  | EmptyStatement
  | IfStatement of expr * statement * statement
  | WhileStatement of expr * statement
  | ForStatement of iterator list * statement
  | PforStatement of iterator list * statement
  | FunctionDecl of datatype * ident * decl list * statement list
  | ForwardDecl of datatype * ident * decl list
  | ReturnStatement of expr
  | VoidReturnStatement
  | SyncStatement
